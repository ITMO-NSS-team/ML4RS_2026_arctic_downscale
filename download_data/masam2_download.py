import os

import pandas as pd
import requests


def masam2_download(folder):
    """
    Download monthly MASAM2 NetCDF files into a year-based folder tree.

    Args:
        folder: Root directory for downloaded files.
    """
    dates_range = pd.date_range("20141101", "20251231")
    for date in dates_range:
        month = date.strftime("%m")
        year = date.strftime("%Y")

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
    masam2_download("D:/MASAM2")
