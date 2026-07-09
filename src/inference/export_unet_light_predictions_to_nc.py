import argparse
import calendar
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import LIGHT_UNET_PREDICTIONS_DIR, MASAM2_LANDMASK_PATH, MASAM2_RAW_DIR


ICE_VARIABLE_NAME = "Sea_Ice_Concentration"
LAND_MASK_VALUE = 120


def validate_date(value):
    """
    Validate a CLI date string.

    Args:
        value: Date in YYYYMMDD format.

    Returns:
        The validated date string.
    """
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise argparse.ArgumentTypeError("Date must use YYYYMMDD format") from error
    return value


def iter_dates(start_date, end_date):
    """
    Yield all dates in an inclusive date range.

    Args:
        start_date: First date in YYYYMMDD format.
        end_date: Last date in YYYYMMDD format.

    Yields:
        Date strings in YYYYMMDD format.
    """
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    if current > end:
        raise ValueError("Start date must not be later than end date")

    while current <= end:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def find_prediction_path(prediction_dir, date):
    """
    Find a Light U-Net prediction matrix for a date.

    Args:
        prediction_dir: Directory with saved .npy predictions.
        date: Date in YYYYMMDD format.

    Returns:
        Path to the prediction matrix.
    """
    prediction_path = prediction_dir / f"{date}.npy"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    return prediction_path


def find_donor_path(masam2_raw_dir, date, donor_path=None):
    """
    Find a MASAM2 NetCDF donor file.

    Args:
        masam2_raw_dir: Root directory with raw MASAM2 NetCDF files.
        date: Date in YYYYMMDD format.
        donor_path: Optional explicit donor file.

    Returns:
        Path to a donor NetCDF file.
    """
    if donor_path is not None:
        if not donor_path.exists():
            raise FileNotFoundError(f"Donor file not found: {donor_path}")
        return donor_path

    year = date[:4]
    year_month = date[:6]
    monthly_donor = masam2_raw_dir / year / f"masam2.{year_month}.nc"
    if monthly_donor.exists():
        return monthly_donor

    donors = sorted(masam2_raw_dir.rglob("masam2.*.nc"))
    if not donors:
        raise FileNotFoundError(f"No MASAM2 donor NetCDF files found in {masam2_raw_dir}")
    return donors[0]


def prediction_to_donor_orientation(prediction):
    """
    Rotate a prediction matrix back to the MASAM2 NetCDF orientation.

    The MASAM2 converter creates .npy matrices with:
    np.rot90(raw, k=1) followed by np.flip(..., axis=0).
    This function applies the inverse orientation transform without changing
    concentration values.

    Args:
        prediction: Two-dimensional prediction matrix.

    Returns:
        Prediction matrix oriented as the donor NetCDF grid.
    """
    return np.rot90(np.flip(prediction, axis=0), k=-1).astype(np.float32)


def load_land_mask(landmask_path):
    """
    Load a MASAM2 land mask in the prediction matrix orientation.

    Args:
        landmask_path: Path to the MASAM2 land mask .npy file.

    Returns:
        Boolean mask where True marks land pixels.
    """
    landmask_path = Path(landmask_path)
    if not landmask_path.exists():
        raise FileNotFoundError(f"Land mask file not found: {landmask_path}")

    landmask = np.load(landmask_path)
    return landmask == LAND_MASK_VALUE


def apply_land_mask(prediction, land_mask):
    """
    Zero prediction values over land.

    Args:
        prediction: Prediction matrix in .npy orientation.
        land_mask: Boolean land mask in the same orientation.

    Returns:
        Prediction matrix with land pixels set to zero.
    """
    if prediction.shape != land_mask.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match land mask "
            f"shape {land_mask.shape}"
        )

    masked_prediction = prediction.copy()
    masked_prediction[land_mask] = 0
    return masked_prediction


def copy_variable_attributes(source_variable, target_variable, exclude=None):
    """
    Copy NetCDF variable attributes.

    Args:
        source_variable: Source NetCDF variable.
        target_variable: Target NetCDF variable.
        exclude: Optional set of attribute names to skip.
    """
    exclude = set() if exclude is None else set(exclude)
    for attr_name in source_variable.ncattrs():
        if attr_name not in exclude:
            target_variable.setncattr(attr_name, source_variable.getncattr(attr_name))


def create_daily_netcdf(
    prediction_path,
    donor_path,
    output_path,
    date,
    landmask_path=MASAM2_LANDMASK_PATH,
):
    """
    Create one daily NetCDF file from a Light U-Net prediction.

    Args:
        prediction_path: Path to the prediction .npy file.
        donor_path: Path to a MASAM2 donor NetCDF file.
        output_path: Path where the daily NetCDF file will be saved.
        date: Date in YYYYMMDD format.
        landmask_path: Path to the MASAM2 land mask .npy file.

    Returns:
        Path to the saved NetCDF file.
    """
    prediction = np.load(prediction_path)
    if prediction.ndim != 2:
        raise ValueError(f"Prediction must be two-dimensional: {prediction_path}")

    if landmask_path is not None:
        land_mask = load_land_mask(landmask_path)
        prediction = apply_land_mask(prediction, land_mask)

    oriented_prediction = prediction_to_donor_orientation(prediction)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    date_obj = datetime.strptime(date, "%Y%m%d")
    month_name = calendar.month_abbr[date_obj.month]

    with Dataset(donor_path, "r") as donor, Dataset(output_path, "w", format="NETCDF4") as output:
        donor_ice = donor.variables[ICE_VARIABLE_NAME]
        expected_shape = donor_ice.shape[-2:]
        if oriented_prediction.shape != expected_shape:
            raise ValueError(
                f"Prediction shape after rotation is {oriented_prediction.shape}, "
                f"expected donor grid shape {expected_shape}"
            )

        output.createDimension("t", 1)
        output.createDimension("y", len(donor.dimensions["y"]))
        output.createDimension("x", len(donor.dimensions["x"]))

        longitude = output.createVariable("Longitude", donor.variables["Longitude"].datatype, ("y", "x"))
        copy_variable_attributes(donor.variables["Longitude"], longitude)
        longitude[:, :] = donor.variables["Longitude"][:, :]

        latitude = output.createVariable("Latitude", donor.variables["Latitude"].datatype, ("y", "x"))
        copy_variable_attributes(donor.variables["Latitude"], latitude)
        latitude[:, :] = donor.variables["Latitude"][:, :]

        day = output.createVariable("Day_of_Month", "i4", ("t",))
        copy_variable_attributes(donor.variables["Day_of_Month"], day)
        day.units = "day_of_month"
        day.setncattr("long-name", f"Day_of_{month_name}_{date_obj.year}")
        day[:] = [date_obj.day]

        ice = output.createVariable(
            ICE_VARIABLE_NAME,
            "f4",
            ("t", "y", "x"),
            zlib=True,
            complevel=5,
            shuffle=True,
            chunksizes=(1, max(1, expected_shape[0] // 2), max(1, expected_shape[1] // 2)),
        )
        ice.units = "1"
        ice.setncattr("long-name", donor_ice.getncattr("long-name"))
        ice.key = "0-1=predicted sea ice concentration fraction"
        ice[0, :, :] = oriented_prediction

    return output_path


def export_predictions_to_netcdf(
    start_date,
    end_date,
    prediction_dir=LIGHT_UNET_PREDICTIONS_DIR,
    masam2_raw_dir=MASAM2_RAW_DIR,
    output_dir=None,
    donor_path=None,
    landmask_path=MASAM2_LANDMASK_PATH,
    overwrite=False,
):
    """
    Export Light U-Net prediction matrices to daily NetCDF files.

    Args:
        start_date: First date in YYYYMMDD format.
        end_date: Last date in YYYYMMDD format.
        prediction_dir: Directory with Light U-Net .npy predictions.
        masam2_raw_dir: Directory with raw MASAM2 donor NetCDF files.
        output_dir: Directory for daily NetCDF outputs.
        donor_path: Optional explicit donor NetCDF path.
        landmask_path: Path to the MASAM2 land mask .npy file.
        overwrite: Whether to overwrite existing files.

    Returns:
        List of saved NetCDF paths.
    """
    output_dir = output_dir or (prediction_dir / "netcdf")
    saved_paths = []

    for date in iter_dates(start_date, end_date):
        prediction_path = find_prediction_path(prediction_dir, date)
        donor = find_donor_path(masam2_raw_dir, date, donor_path=donor_path)
        output_path = output_dir / date[:4] / f"unet_light_{date}.nc"

        if output_path.exists() and not overwrite:
            print(f"Skipping existing file: {output_path}")
            saved_paths.append(output_path)
            continue

        saved_path = create_daily_netcdf(
            prediction_path=prediction_path,
            donor_path=donor,
            output_path=output_path,
            date=date,
            landmask_path=landmask_path,
        )
        print(f"Saved NetCDF: {saved_path}")
        saved_paths.append(saved_path)

    return saved_paths


def main():
    """Parse CLI arguments and export prediction matrices to NetCDF."""
    parser = argparse.ArgumentParser(
        description="Export Light U-Net prediction matrices to daily NetCDF files."
    )
    parser.add_argument("--start", required=True, type=validate_date)
    parser.add_argument("--end", required=True, type=validate_date)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=LIGHT_UNET_PREDICTIONS_DIR,
        help="Directory with saved Light U-Net .npy predictions.",
    )
    parser.add_argument(
        "--masam2-raw-dir",
        type=Path,
        default=MASAM2_RAW_DIR,
        help="Directory with raw MASAM2 donor NetCDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for daily NetCDF files. Defaults to prediction-dir/netcdf.",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        default=None,
        help="Explicit donor NetCDF file. Defaults to a matching MASAM2 month if available.",
    )
    parser.add_argument(
        "--landmask",
        type=Path,
        default=MASAM2_LANDMASK_PATH,
        help="MASAM2 land mask .npy file. Land pixels are set to zero before export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NetCDF files.",
    )

    args = parser.parse_args()

    export_predictions_to_netcdf(
        start_date=args.start,
        end_date=args.end,
        prediction_dir=args.prediction_dir,
        masam2_raw_dir=args.masam2_raw_dir,
        output_dir=args.output_dir,
        donor_path=args.donor,
        landmask_path=args.landmask,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
