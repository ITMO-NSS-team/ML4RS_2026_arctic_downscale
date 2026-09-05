import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (
    LIGHT_UNET_PREDICTIONS_DIR,
    MASAM2_LANDMASK_PATH,
    MASAM2_RAW_DIR,
    OSISAF_DIR,
)
from inference.export_unet_light_predictions_to_nc import (
    export_predictions_to_netcdf,
    iter_dates,
    validate_date,
)
from inference.predict_osisaf_by_date import predict_osisaf_by_date


SUPPORTED_MODEL = "unet_light"


def predict_osisaf_range_to_netcdf(
    start_date,
    end_date,
    model_name=SUPPORTED_MODEL,
    weights_path=None,
    osisaf_dir=OSISAF_DIR,
    prediction_dir=LIGHT_UNET_PREDICTIONS_DIR,
    masam2_raw_dir=MASAM2_RAW_DIR,
    output_dir=None,
    donor_path=None,
    landmask_path=MASAM2_LANDMASK_PATH,
    overwrite=False,
):
    """
    Run Light U-Net predictions from OSISAF for a date range and export them to NetCDF.

    Args:
        start_date: First date in YYYYMMDD format.
        end_date: Last date in YYYYMMDD format.
        model_name: Model name. Only unet_light is supported by this NetCDF export.
        weights_path: Optional path to Light U-Net weights or checkpoint.
        osisaf_dir: Directory with OSISAF .npy inputs.
        prediction_dir: Directory for intermediate .npy predictions.
        masam2_raw_dir: Directory with raw MASAM2 donor NetCDF files.
        output_dir: Directory for daily NetCDF outputs.
        donor_path: Optional explicit donor NetCDF path.
        landmask_path: Path to the MASAM2 land mask .npy file.
        overwrite: Whether to overwrite existing NetCDF files.

    Returns:
        Tuple with saved prediction paths and saved NetCDF paths.
    """
    if model_name != SUPPORTED_MODEL:
        raise ValueError(f"This script supports only {SUPPORTED_MODEL}")

    prediction_dir = Path(prediction_dir)
    dates = list(iter_dates(start_date, end_date))

    prediction_paths = []
    for date in dates:
        prediction_path = predict_osisaf_by_date(
            model_name=model_name,
            date=date,
            weights_path=weights_path,
            output_dir=prediction_dir,
            osisaf_dir=osisaf_dir,
        )
        prediction_paths.append(prediction_path)

    netcdf_paths = export_predictions_to_netcdf(
        start_date=start_date,
        end_date=end_date,
        prediction_dir=prediction_dir,
        masam2_raw_dir=masam2_raw_dir,
        output_dir=output_dir,
        donor_path=donor_path,
        landmask_path=landmask_path,
        overwrite=overwrite,
    )

    return prediction_paths, netcdf_paths


def main():
    """Parse CLI arguments and run OSISAF-to-NetCDF Light U-Net inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Light U-Net OSISAF-only predictions for a date range and "
            "export the results to MASAM2-like NetCDF files."
        )
    )
    parser.add_argument(
        "--model",
        choices=(SUPPORTED_MODEL,),
        required=True,
        help="Model architecture. This pipeline currently supports only unet_light.",
    )
    parser.add_argument("--start", required=True, type=validate_date)
    parser.add_argument("--end", required=True, type=validate_date)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to .pth weights. Defaults to outputs/models/unet_light_final.pth.",
    )
    parser.add_argument(
        "--osisaf-dir",
        type=Path,
        default=OSISAF_DIR,
        help="Path to OSISAF .npy directory.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=LIGHT_UNET_PREDICTIONS_DIR,
        help="Directory for intermediate Light U-Net .npy predictions.",
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
        help=(
            "Explicit donor NetCDF file. Defaults to a matching MASAM2 month "
            "if available."
        ),
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

    prediction_paths, netcdf_paths = predict_osisaf_range_to_netcdf(
        start_date=args.start,
        end_date=args.end,
        model_name=args.model,
        weights_path=args.weights,
        osisaf_dir=args.osisaf_dir,
        prediction_dir=args.prediction_dir,
        masam2_raw_dir=args.masam2_raw_dir,
        output_dir=args.output_dir,
        donor_path=args.donor,
        landmask_path=args.landmask,
        overwrite=args.overwrite,
    )

    print(f"Saved predictions: {len(prediction_paths)}")
    print(f"Saved NetCDF files: {len(netcdf_paths)}")


if __name__ == "__main__":
    main()
