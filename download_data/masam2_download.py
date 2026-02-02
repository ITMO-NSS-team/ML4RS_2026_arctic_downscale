import os

import pandas as pd
import requests


def masam2_download(folder):
    dates_range = pd.date_range("20120101", "20261231", freq="MS")
    for date in dates_range:
        month = date.strftime("%m")
        year = date.strftime("%Y")

        url = f"https://noaadata.apps.nsidc.org/NOAA/G10005/Data/{year}/masam2.{year}{month}.nc.gz"
        filename = f"masam2.{year}{month}.nc.gz"
        year_dir = os.path.join(folder, str(year))
        os.makedirs(year_dir, exist_ok=True)
        filepath = os.path.join(year_dir, filename)

        # Skip if file exists
        if os.path.exists(filepath):
            print(f"Skipping existing file: {filename}")
            continue

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
    masam2_download("data")
