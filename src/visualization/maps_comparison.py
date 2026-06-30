import sys
from pathlib import Path

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

    masam2_mask = np.load(masam2_landmask_path)
    mask_rgba = np.zeros((*masam2_mask.shape, 4), dtype=np.float32)
    mask_rgba[masam2_mask == 120] = [0.8, 0.7, 0.7, 1]
    mask_rgba = mask_rgba[row_start:row_end, col_start:col_end]

    dates = pd.date_range(start_date, end_date)
    dates = [date.strftime("%Y%m%d") for date in dates]
    for date in dates:
        try:
            print(date)
            osisaf_matrix = np.load(osisaf_dir / f"{date}.npy")
            masie_matrix = np.load(masie_dir / f"{date}.npy")
            hybrid_matrix = np.load(hybrid_dir / f"{date}.npy")
            masam2_matrix = np.load(masam2_dir / f"{date}.npy")

            light_unet = np.load(light_unet_dir / f"{date}.npy")
            attention_unet = np.load(attention_unet_dir / f"{date}.npy")

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
