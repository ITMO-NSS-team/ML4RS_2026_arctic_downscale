import os

import matplotlib

matplotlib.use("Agg")
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MASAM2_DIR, OSISAF_DIR
from dataset import IceConcentrationDataset


def visualize_14day_pairs(
    osisaf_dir=OSISAF_DIR,
    masam2_dir=MASAM2_DIR,
    output_dir="14day_visualizations",
    step_days=14,
    max_samples=None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        print(f"Loading a dataset...")
        dataset = IceConcentrationDataset(osisaf_dir=osisaf_dir, masam2_dir=masam2_dir)

        print(f"The dataset is loaded. Total pairs: {len(dataset)}")

        stats = dataset.get_statistics()
        print(f"Date range: {stats['date_range'][0]} - {stats['date_range'][1]}")

    except Exception as e:
        print(f"Load dataset exception: {e}")
        return

    print("Extracting dates from files...")
    dates = []
    date_to_idx = {}

    for idx, (osisaf_path, masam2_path) in enumerate(dataset.pairs):
        try:
            filename = os.path.basename(osisaf_path)
            date_str = None

            for i in range(len(filename) - 7):
                if filename[i : i + 8].isdigit() and len(filename[i : i + 8]) == 8:
                    date_str = filename[i : i + 8]
                    year = int(date_str[0:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                        break
                    else:
                        date_str = None

            if date_str:
                dates.append(date_str)
                date_to_idx[date_str] = idx
        except Exception as e:
            print(f"File processing error {osisaf_path}: {e}")
            continue

    if not dates:
        print("Valid dates not found!")
        return

    print(f"Found {len(dates)} dates")

    dates.sort()

    date_objects = []
    for date_str in dates:
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            date_objects.append((date_str, date_obj))
        except Exception as e:
            print(f"Couldn't convert the date {date_str}: {e}")

    if not date_objects:
        print("No valid dates found after conversion!")
        return

    print(f"Successfully converted {len(date_objects)} dates")

    selected_dates = []
    if date_objects:
        current_date = date_objects[0][1]
        selected_dates.append(date_objects[0])

        for date_str, date_obj in date_objects[1:]:
            if (date_obj - current_date).days >= step_days:
                selected_dates.append((date_str, date_obj))
                current_date = date_obj

    if max_samples and len(selected_dates) > max_samples:
        selected_dates = selected_dates[:max_samples]

    print(f"Will be visualized {len(selected_dates)} dates with step {step_days} days")

    if not selected_dates:
        print("There are no dates for visualization!")
        return

    for i, (date_str, date_obj) in enumerate(selected_dates):
        try:
            print(f"Processing {i + 1}/{len(selected_dates)}: {date_str}")

            idx = date_to_idx.get(date_str)
            if idx is None:
                print(f"Index not found for date {date_str}, skip...")
                continue

            sample = dataset[idx]

            lr_data = sample["lr"].squeeze().cpu().numpy()
            hr_data = sample["hr"].squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            formatted_date = date_obj.strftime("%Y-%m-%d")

            im1 = axes[0].imshow(lr_data, cmap="viridis", aspect="auto")
            axes[0].set_title(
                f"OSISAF - {formatted_date}", fontsize=14, fontweight="bold"
            )
            axes[0].set_xlabel("Latitude")
            axes[0].set_ylabel("Longitude")
            plt.colorbar(
                im1, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04
            )

            im2 = axes[1].imshow(hr_data, cmap="viridis", aspect="auto")
            axes[1].set_title(
                f"MASAM2 - {formatted_date}", fontsize=14, fontweight="bold"
            )
            axes[1].set_xlabel("Latitude")
            axes[1].set_ylabel("Longitude")
            plt.colorbar(
                im2, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04
            )

            plt.suptitle(
                f"Sea ice concentration - {formatted_date}",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )

            plt.tight_layout()

            output_path = os.path.join(output_dir, f"ice_concentration_{date_str}.png")
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"Date processing error {date_str}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\nReady! All images are saved to a directory: {output_dir}")
    print(f"Total saved: {len(selected_dates)} images")


if __name__ == "__main__":
    visualize_14day_pairs(
        osisaf_dir=OSISAF_DIR,
        masam2_dir=MASAM2_DIR,
        output_dir="ice_concentration_pairs",
        step_days=1,
    )
