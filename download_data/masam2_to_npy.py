import os
import shutil
import tempfile
import warnings

import numpy as np
import xarray as xr
from scipy import interpolate

warnings.filterwarnings("ignore")


class MASAM2_to_NPY_Converter:
    """
    Convert from MASAM2 (.nc.gz) to .npy
    """

    def __init__(self, masam2_dir, output_dir=None, temp_dir=None):
        """
        Args:
            masam2_dir: Directory with MASAM2 files
            output_dir: Directory for .npy files
            temp_dir: Directory for temporal files
        """
        self.masam2_dir = masam2_dir
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(masam2_dir), "MASAM2_npy")
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        if temp_dir is None:
            self.temp_dir = os.path.join(tempfile.gettempdir(), "masam2_temp")
        else:
            self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        self.stats = {
            "total_files": 0,
            "total_frames": 0,
            "converted_frames": 0,
            "failed_files": [],
        }

    def read_masam2_file(self, file_path):
        """
        return: xarray Dataset
        """
        print(f"Reading file: {os.path.basename(file_path)}")

        try:
            engines_to_try = ["h5netcdf", "netcdf4"]

            for engine in engines_to_try:
                try:
                    ds = xr.open_dataset(file_path, engine=engine)
                    return ds
                except:
                    continue

            temp_file = os.path.join(
                self.temp_dir, os.path.basename(file_path).replace(".gz", "")
            )
            if file_path.endswith(".gz"):
                import gzip

                with gzip.open(file_path, "rb") as f_in:
                    with open(temp_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                for engine in engines_to_try:
                    try:
                        ds = xr.open_dataset(temp_file, engine=engine)
                        self._temp_file_to_delete = temp_file
                        return ds
                    except:
                        continue

                if os.path.exists(temp_file):
                    os.remove(temp_file)

            raise Exception("Couldn't open the file using any of the methods")

        except Exception as e:
            print(f"File reading error: {e}")
            return None

    def convert_masam2_file(self, file_path, output_subdir=None):
        """
        Converts MASAM2 file to .npy file
        """
        print(f"\n{'=' * 60}")
        print(f"CONVERSION: {os.path.basename(file_path)}")
        print(f"{'=' * 60}")

        ds = None
        try:
            ds = self.read_masam2_file(file_path)
            if ds is None:
                self.stats["failed_files"].append(os.path.basename(file_path))
                return

            if "Sea_Ice_Concentration" not in ds and "sea ice concentration" not in ds:
                print("There is no variable in the file 'sea ice concentration'")
                self.stats["failed_files"].append(os.path.basename(file_path))
                return

            if "Sea_Ice_Concentration" in ds:
                ice_data = ds["Sea_Ice_Concentration"].values
            if "sea ice concentration" in ds:
                ice_data = ds["sea ice concentration"].values

            ice_data = np.rot90(ice_data, k=1, axes=(1, 2))
            ice_data = np.flip(ice_data, axis=1)

            if "Day of Month" in ds:
                days = ds["Day of Month"].values
            else:
                days = np.arange(1, ice_data.shape[0] + 1)

            print(f"The data is uploaded:")
            print(f"   Shape: {ice_data.shape} (days×height×width)")
            print(f"   Data type: {ice_data.dtype}")
            print(f"   Days: {days}")

            file_name = os.path.basename(file_path)
            parts = file_name.split(".")
            year_month = parts[1]
            year = year_month[:4]
            month = year_month[4:6]

            self.stats["total_files"] += 1
            self.stats["total_frames"] += ice_data.shape[0]

            if output_subdir is None:
                base_name = f"{year}_{month}"
                output_subdir = os.path.join(self.output_dir, base_name)
            else:
                output_subdir = os.path.join(self.output_dir, output_subdir)

            os.makedirs(output_subdir, exist_ok=True)

            print("\n Saving files .npy...")

            for i, day_num in enumerate(days):
                day_data = ice_data[i]

                day_data_processed = self.process_values(day_data)

                day_str = f"{int(day_num):02d}"
                output_filename = f"masam2_{year}{month}{day_str}.npy"
                output_path = os.path.join(output_subdir, output_filename)
                np.save(output_path, day_data_processed)

                all_days_dir = os.path.join(self.output_dir, "all_days")
                os.makedirs(all_days_dir, exist_ok=True)
                all_days_path = os.path.join(all_days_dir, output_filename)
                np.save(all_days_path, day_data_processed)

                print(
                    f"   Day {day_str}: {output_filename} ({day_data_processed.shape}, {day_data_processed.dtype})"
                )
                self.stats["converted_frames"] += 1

            print(f"\n The file has been successfully converted!")
            print(f"   Files are saved in: {output_subdir}")
            print(f"   Also in: {os.path.join(self.output_dir, 'all_days')}")

        except Exception as e:
            print(f" File conversion error {file_path}: {e}")
            import traceback

            traceback.print_exc()
            self.stats["failed_files"].append(os.path.basename(file_path))

        finally:
            if ds is not None:
                ds.close()

            if hasattr(self, "_temp_file_to_delete") and os.path.exists(
                self._temp_file_to_delete
            ):
                try:
                    os.remove(self._temp_file_to_delete)
                    delattr(self, "_temp_file_to_delete")
                except:
                    pass

    def process_values(self, data):
        cleaned = data.copy().astype(np.int16)
        cleaned[(cleaned == 104) | (cleaned == 119) | (cleaned == 120)] = 0
        mask = cleaned == 110

        result = interpolate_missing_pixels(cleaned, mask)

        cleaned[mask] = result[mask]

        return cleaned

    def batch_convert(self, file_pattern="masam2.*.nc"):
        """
        Batch conversion of all MASAM2 files in the directory
        """
        print(f"\n{'=' * 60}")
        print(f"BATCH CONVERSION MASAM2 FILES")
        print(f"Input directory: {self.masam2_dir}")
        print(f"Target directory: {self.output_dir}")
        print(f"Temporal directory: {self.temp_dir}")
        print(f"{'=' * 60}")

        masam2_files = []
        for root, dirs, files in os.walk(self.masam2_dir):
            for file in files:
                if (
                    file.endswith(".nc.gz") or file.endswith(".nc")
                ) and "masam2" in file.lower():
                    masam2_files.append(os.path.join(root, file))

        print(f" Files found MASAM2: {len(masam2_files)}")

        if len(masam2_files) == 0:
            print("️  Files MASAM2 not found!")
            print("   Expected format: masam2.YYYYMM.nc.gz или masam2.YYYYMM.nc")
            return

        print(f"\n STARTING THE CONVERSION...")

        for i, file_path in enumerate(masam2_files):
            print(f"\n[{i + 1}/{len(masam2_files)}] ", end="")
            self.convert_masam2_file(file_path)


def quick_convert(masam2_file, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(masam2_file), "npy_output")

    converter = MASAM2_to_NPY_Converter(
        masam2_dir=os.path.dirname(masam2_file), output_dir=output_dir
    )
    converter.convert_masam2_file(masam2_file)

    npy_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".npy") and file.startswith("masam2_"):
                npy_files.append(os.path.join(root, file))

    if npy_files:
        print(f"\nTHE FIRST FILE CREATED:")
        sample = np.load(npy_files[0])
        print(f"   Name: {os.path.basename(npy_files[0])}")
        print(f"   Shape: {sample.shape}")
        print(f"   Type: {sample.dtype}")


def interpolate_missing_pixels(
    image: np.ndarray, mask: np.ndarray, method: str = "nearest", fill_value: int = 0
):
    """
    Fill missing values from mask with interpolation

    Args:
        image: a 2D image
        mask: a 2D boolean image, True indicates missing values
        method: interpolation method, one of 'nearest', 'linear', 'cubic'.
        fill_value: which value to use for filling up data outside the convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
        :return: the image with missing values interpolated
    """
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]
    interp_values = interpolate.griddata(
        (known_x, known_y),
        known_v,
        (missing_x, missing_y),
        method=method,
        fill_value=fill_value,
    )
    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values
    return interp_image


if __name__ == "__main__":
    print("Converter MASAM2 → .npy format")

    for year in ["20" + str(i) for i in range(20, 21)]:
        for month in [str(j) for j in range(1, 13)]:
            if len(month) == 1:
                month = "0" + month

            filename = f"masam2.{year}{month}.nc"
            input_file = f"D:/MASAM2/{year}/{filename}"

            if os.path.exists(input_file):
                print(f"File found: {input_file}")
                output_dir = os.path.join(
                    os.path.dirname(os.path.dirname(input_file)), "MASAM2_npy"
                )
                print(f" The output directory: {output_dir}")

                quick_convert(input_file, output_dir)

            else:
                print(f" File not found: {input_file}")
