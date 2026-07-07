import os
import sys
from datetime import datetime
from pathlib import Path

import requests

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import MASAM2_RAW_DIR


def iter_months(start_year_month, end_year_month):
    """Yield YYYY, MM pairs for all months in an inclusive YYYYMM range."""
    current = datetime.strptime(start_year_month, "%Y%m")
    end = datetime.strptime(end_year_month, "%Y%m")
    while current <= end:
        yield current.strftime("%Y"), current.strftime("%m")
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def masam2_download(folder, start_year_month, end_year_month):
    """
    Download monthly MASAM2 NetCDF files into a year-based folder tree.

    Args:
        folder: Root directory for downloaded files.
        start_year_month: First month in YYYYMM format.
        end_year_month: Last month in YYYYMM format.
    """
    for year, month in iter_months(start_year_month, end_year_month):
        url = f"https://noaadata.apps.nsidc.org/NOAA/G10005/Data/{year}/masam2.{year}{month}.nc"
        filename = f"masam2.{year}{month}.nc"
        filepath = os.path.join(folder, str(year), filename)

        if os.path.exists(filepath):
            print(f"Skipping existing file: {filename}")
            continue
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(response.content)

            print(f"Downloaded {filename}")

        except Exception as e:
            print(f"Failed to download {filename}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download monthly MASAM2 NetCDF files.")
    parser.add_argument("--start", required=True, help="First month in YYYYMM format.")
    parser.add_argument("--end", required=True, help="Last month in YYYYMM format.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MASAM2_RAW_DIR,
        help="Root directory for downloaded MASAM2 NetCDF files.",
    )
    args = parser.parse_args()

    masam2_download(args.output_dir, args.start, args.end)
