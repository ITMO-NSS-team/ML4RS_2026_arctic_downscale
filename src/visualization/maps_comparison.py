import sys
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (
    ATTENTION_UNET_PREDICTIONS_DIR,
    LIGHT_UNET_PREDICTIONS_DIR,
    MAP_COMPARISON_FIGURES_DIR,
    MASAM2_LANDMASK_PATH,
    MASAM2_DIR,
    MASIE_REPROJECTED_DIR,
    MANUAL_HYBRID_REPROJECTED_DIR,
    OSISAF_REPROJECTED_DIR,
)


def extract_date_from_filename(filename):
    """Extract an YYYYMMDD date from a filename."""
    match = re.search(r"(\d{8})", filename)
    if match is None:
        return None
    return match.group(1)


def index_npy_files_by_date(directory):
    """Index .npy files recursively by date from filename."""
    files_by_date = {}
    for path in sorted(Path(directory).rglob("*.npy")):
        date = extract_date_from_filename(path.name)
        if date is not None:
            files_by_date.setdefault(date, path)
    return files_by_date


def load_by_date(files_by_date, date, label):
    """Load one date from a recursively indexed .npy directory."""
    path = files_by_date.get(date)
    if path is None:
        raise FileNotFoundError(f"{label} file not found for date {date}")
    return np.load(path)


def compare_prediction_maps(
    start_date="20240901",
    end_date="20260301",
    light_unet_dir=LIGHT_UNET_PREDICTIONS_DIR,
    attention_unet_dir=ATTENTION_UNET_PREDICTIONS_DIR,
    osisaf_dir=OSISAF_REPROJECTED_DIR,
    masie_dir=MASIE_REPROJECTED_DIR,
    hybrid_dir=MANUAL_HYBRID_REPROJECTED_DIR,
    masam2_dir=MASAM2_DIR,
    masam2_landmask_path=MASAM2_LANDMASK_PATH,
    output_dir=MAP_COMPARISON_FIGURES_DIR,
    row_start=1100,
    row_end=1500,
    col_start=1050,
    col_end=1550,
):
    """
    Save side-by-side map comparisons for ground truth, baselines, and U-Net outputs.

    Args:
        start_date: First date in YYYYMMDD format.
        end_date: Last date in YYYYMMDD format.
        light_unet_dir: Directory with Light U-Net prediction .npy files.
        attention_unet_dir: Directory with Attention U-Net prediction .npy files.
        osisaf_dir: Directory with reprojected OSISAF .npy files.
        masie_dir: Directory with reprojected MASIE .npy files.
        hybrid_dir: Directory with manual hybrid .npy files.
        masam2_dir: Directory with reprojected MASAM2 .npy files.
        masam2_landmask_path: Path to the MASAM2 land mask .npy file.
        output_dir: Directory for saved comparison figures.
        row_start: First row of the cropped region.
        row_end: Last row of the cropped region.
        col_start: First column of the cropped region.
        col_end: Last column of the cropped region.
    """
    light_unet_dir = Path(light_unet_dir)
    attention_unet_dir = Path(attention_unet_dir)
    osisaf_dir = Path(osisaf_dir)
    masie_dir = Path(masie_dir)
    hybrid_dir = Path(hybrid_dir)
    masam2_dir = Path(masam2_dir)
    masam2_landmask_path = Path(masam2_landmask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    light_unet_by_date = index_npy_files_by_date(light_unet_dir)
    attention_unet_by_date = index_npy_files_by_date(attention_unet_dir)
    osisaf_by_date = index_npy_files_by_date(osisaf_dir)
    masie_by_date = index_npy_files_by_date(masie_dir)
    hybrid_by_date = index_npy_files_by_date(hybrid_dir)
    masam2_by_date = index_npy_files_by_date(masam2_dir)

    masam2_mask = np.load(masam2_landmask_path)
    mask_rgba = np.zeros((*masam2_mask.shape, 4), dtype=np.float32)
    mask_rgba[masam2_mask == 120] = [0.8, 0.7, 0.7, 1]
    mask_rgba = mask_rgba[row_start:row_end, col_start:col_end]

    dates = pd.date_range(start_date, end_date)
    dates = [date.strftime("%Y%m%d") for date in dates]
    for date in dates:
        try:
            print(date)
            osisaf_matrix = load_by_date(osisaf_by_date, date, "OSISAF")
            masie_matrix = load_by_date(masie_by_date, date, "MASIE")
            hybrid_matrix = load_by_date(hybrid_by_date, date, "Manual hybrid")
            masam2_matrix = load_by_date(masam2_by_date, date, "MASAM2")

            light_unet = load_by_date(light_unet_by_date, date, "Light U-Net")
            attention_unet = load_by_date(
                attention_unet_by_date,
                date,
                "Attention U-Net",
            )

            osisaf_matrix = osisaf_matrix[row_start:row_end, col_start:col_end]
            masie_matrix = masie_matrix[row_start:row_end, col_start:col_end]
            hybrid_matrix = hybrid_matrix[row_start:row_end, col_start:col_end]
            masam2_matrix = masam2_matrix[row_start:row_end, col_start:col_end]
            light_unet = light_unet[row_start:row_end, col_start:col_end]
            attention_unet = attention_unet[row_start:row_end, col_start:col_end]

            plt.rcParams["figure.figsize"] = (19, 4)
            fig, axs = plt.subplots(nrows=1, ncols=5)
            axs[0].imshow(masam2_matrix, cmap="Blues_r", interpolation="none")
            axs[0].imshow(mask_rgba)
            axs[0].set_title("MASAM2 - Ground Truth")

            axs[1].imshow(osisaf_matrix, cmap="Blues_r", interpolation="none")
            axs[1].imshow(mask_rgba)
            axs[1].contour(masie_matrix, levels=[0], colors="red", linewidths=0.5)
            axs[1].set_title(
                f"MAE={np.mean(abs(osisaf_matrix - masam2_matrix)):.03f}\n"
                "OSISAF (nearest) + MASIE contour"
            )

            axs[2].imshow(hybrid_matrix, cmap="Blues_r", interpolation="none")
            axs[2].imshow(mask_rgba)
            axs[2].contour(masie_matrix, levels=[0], colors="red", linewidths=0.5)
            axs[2].set_title(
                f"MAE={np.mean(abs(hybrid_matrix - masam2_matrix)):.03f}\n"
                "Manual Hybrid + MASIE contour"
            )

            axs[3].imshow(light_unet, cmap="Blues_r", interpolation="none")
            axs[3].imshow(mask_rgba)
            axs[3].contour(masie_matrix, levels=[0], colors="red", linewidths=0.5)
            axs[3].set_title(
                f"MAE={np.mean(abs(light_unet - masam2_matrix)):.03f}\n"
                "Light U-Net + MASIE contour"
            )

            axs[4].imshow(attention_unet, cmap="Blues_r", interpolation="none")
            axs[4].imshow(mask_rgba)
            axs[4].contour(masie_matrix, levels=[0], colors="red", linewidths=0.5)
            axs[4].set_title(
                f"MAE={np.mean(abs(attention_unet - masam2_matrix)):.03f}\n"
                "Attention U-Net + MASIE contour"
            )

            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

            for spine in axs[0].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

            for spine in axs[3].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("green")
                spine.set_linewidth(3)

            for spine in axs[4].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("green")
                spine.set_linewidth(4)

            plt.suptitle(date)
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{date}_predictions.png",
                transparent=True,
                bbox_inches="tight",
            )
            plt.close(fig)

        except Exception as error:
            print(error)


if __name__ == "__main__":
    compare_prediction_maps()
