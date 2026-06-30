import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (
    MASAM2_MASK_NETCDF_PATH,
    MASIE_TARGET_GRID_PATH,
    OSISAF_DIR,
    OSISAF_REPROJECT_CROP_BOUNDS,
    OSISAF_REPROJECTED_DIR,
)


def extract_date_from_filename(filename):
    """Extract an YYYYMMDD date from a filename."""
    match = re.search(r"(\d{8})", filename)
    if match is None:
        raise ValueError(f"No YYYYMMDD date found in filename: {filename}")
    return match.group(1)


def convert_to_wsl_path(path):
    """Convert a Windows path to a WSL /mnt/<drive>/ path for CDO."""
    path = Path(path).resolve()
    path_text = str(path)
    drive = path.drive.rstrip(":").lower()
    if not drive:
        return path_text.replace("\\", "/")

    relative_path = path_text[len(path.drive) :].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{relative_path}"


def load_reprojected_ice_concentration(path):
    """Load the ice_conc variable from a temporary CDO output file."""
    with nc.Dataset(path) as dataset:
        if "ice_conc" not in dataset.variables:
            raise KeyError(f"Variable 'ice_conc' not found in {path}")

        ice_variable = dataset.variables["ice_conc"]
        if ice_variable.ndim == 3:
            data = ice_variable[0, :, :]
        else:
            data = ice_variable[:, :]

    return np.asarray(data, dtype=np.float32)


def load_mask(mask_path):
    """Load the MASAM2 land mask in the pre-rotation OSISAF crop orientation."""
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    if mask_path.suffix == ".npy":
        mask_data = np.load(mask_path)
    else:
        with nc.Dataset(mask_path) as dataset:
            if "sea ice concentration" in dataset.variables:
                mask_variable = dataset.variables["sea ice concentration"]
            else:
                for variable_name in ("sea_ice_concentration", "ice_conc", "mask"):
                    if variable_name in dataset.variables:
                        mask_variable = dataset.variables[variable_name]
                        print(
                            f"Using variable '{variable_name}' instead of "
                            "'sea ice concentration'"
                        )
                        break
                else:
                    raise KeyError(
                        "Mask file does not contain 'sea ice concentration', "
                        "'sea_ice_concentration', 'ice_conc', or 'mask'"
                    )

            mask_data = mask_variable[:]

    if mask_data.ndim == 3:
        mask_data = mask_data[0]

    mask_data = np.asarray(mask_data, dtype=np.float64)
    mask_data[mask_data == 119] = 120
    return mask_data == 120


def preprocess_osisaf_matrix(data, land_mask=None, crop_slice=None):
    """Prepare a reprojected OSISAF matrix for comparison with MASAM2."""
    if crop_slice is not None:
        data = data[crop_slice]

    print(f"Cropped data shape: {data.shape}")
    data = np.maximum(data, 0)

    if land_mask is not None:
        if data.shape != land_mask.shape:
            raise ValueError(
                f"OSISAF matrix shape {data.shape} does not match land mask "
                f"shape {land_mask.shape}. The mask is expected to match the "
                "cropped data before fliplr/rot90."
            )
        data = data.copy()
        data[land_mask] = 0

    data = data / 100.0
    data = np.rot90(np.fliplr(data))
    return data.astype(np.float32)


def save_preview(matrix, preview_path):
    """Save a quick visual check for one reprojected matrix."""
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imshow(matrix, interpolation="none", cmap="Blues_r")
    plt.title("OSISAF")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(preview_path, dpi=100)
    plt.close()


def reproject_single_file(
    input_path,
    target_grid_path,
    output_dir,
    temp_dir,
    land_mask=None,
    crop_slice=None,
    save_previews=True,
):
    """Reproject one OSISAF NetCDF file with CDO and save it as .npy."""
    date = extract_date_from_filename(input_path.name)
    temp_path = temp_dir / f"remap_temp_{date}.nc"

    command = [
        "wsl",
        "-e",
        "cdo",
        "-selname,ice_conc",
        f"-remapnn,{convert_to_wsl_path(target_grid_path)}",
        convert_to_wsl_path(input_path),
        convert_to_wsl_path(temp_path),
    ]

    print("Running:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"CDO failed for {input_path}:\n{result.stderr}")

    try:
        matrix = load_reprojected_ice_concentration(temp_path)
        matrix = preprocess_osisaf_matrix(
            matrix,
            land_mask=land_mask,
            crop_slice=crop_slice,
        )

        output_path = output_dir / f"{date}.npy"
        np.save(output_path, matrix)

        if save_previews:
            save_preview(matrix, output_dir / "preview" / f"{date}.png")

        print(f"Saved: {output_path} ({matrix.shape})")
        return output_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_crop(value):
    """Parse a crop spec in y_start,y_end,x_start,x_end format."""
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Crop must be y_start,y_end,x_start,x_end")

    return crop_bounds_to_slice(parts)


def crop_bounds_to_slice(bounds):
    """Convert y_start,y_end,x_start,x_end bounds to NumPy slices."""
    y_start, y_end, x_start, x_end = bounds
    return (slice(y_start, y_end), slice(x_start, x_end))


def reproject_osisaf_directory(
    input_dir=OSISAF_DIR,
    target_grid_path=MASIE_TARGET_GRID_PATH,
    output_dir=OSISAF_REPROJECTED_DIR,
    landmask_path=MASAM2_MASK_NETCDF_PATH,
    crop_slice=crop_bounds_to_slice(OSISAF_REPROJECT_CROP_BOUNDS),
    save_previews=True,
):
    """Reproject all OSISAF NetCDF files in a directory to MASAM2/MASIE grid."""
    input_dir = Path(input_dir)
    target_grid_path = Path(target_grid_path)
    output_dir = Path(output_dir)
    landmask_path = Path(landmask_path) if landmask_path else None

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_temp_remap"
    temp_dir.mkdir(parents=True, exist_ok=True)

    land_mask = None
    if landmask_path is not None:
        try:
            land_mask = load_mask(landmask_path)
            print(f"Mask loaded: {land_mask.shape}")
        except Exception as error:
            print(f"Mask loading failed: {error}")
            print("Continuing without mask.")

    input_files = sorted(input_dir.glob("*.nc"))
    if not input_files:
        print(f"No .nc files found in {input_dir}")
        return []

    outputs = []
    for index, input_path in enumerate(input_files, start=1):
        print(f"\n--- Processing {index}/{len(input_files)}: {input_path.name} ---")
        try:
            output_path = reproject_single_file(
                input_path=input_path,
                target_grid_path=target_grid_path,
                output_dir=output_dir,
                temp_dir=temp_dir,
                land_mask=land_mask,
                crop_slice=crop_slice,
                save_previews=save_previews,
            )
            outputs.append(output_path)
        except Exception as error:
            print(f"Error: {error}")

    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Reproject OSISAF NetCDF files.")
    parser.add_argument("--input-dir", type=Path, default=OSISAF_DIR)
    parser.add_argument("--target-grid", type=Path, default=MASIE_TARGET_GRID_PATH)
    parser.add_argument("--output-dir", type=Path, default=OSISAF_REPROJECTED_DIR)
    parser.add_argument("--landmask", type=Path, default=MASAM2_MASK_NETCDF_PATH)
    parser.add_argument(
        "--crop",
        type=parse_crop,
        default=crop_bounds_to_slice(OSISAF_REPROJECT_CROP_BOUNDS),
        help="Optional crop as y_start,y_end,x_start,x_end before final rotation.",
    )
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()

    reproject_osisaf_directory(
        input_dir=args.input_dir,
        target_grid_path=args.target_grid,
        output_dir=args.output_dir,
        landmask_path=args.landmask,
        crop_slice=args.crop,
        save_previews=not args.no_previews,
    )


if __name__ == "__main__":
    main()
