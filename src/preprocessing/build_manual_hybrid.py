import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import (
    MANUAL_HYBRID_REPROJECTED_DIR,
    MASIE_REPROJECTED_DIR,
    OSISAF_REPROJECTED_DIR,
)


def build_hybrid_matrix(osisaf_matrix, masie_matrix, threshold=0.15):
    """Combine OSISAF concentration with a MASIE ice/no-ice mask."""
    hybrid_matrix = osisaf_matrix.copy()
    hybrid_matrix[masie_matrix != 1] = 0
    hybrid_matrix[(hybrid_matrix < threshold) & (masie_matrix == 1)] = threshold
    return hybrid_matrix.astype(np.float32)


def save_preview(matrix, preview_path, title):
    """Save a quick visual check for one manual hybrid matrix."""
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    plt.imshow(matrix, interpolation="none", cmap="Blues_r")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(preview_path, dpi=100)
    plt.close()


def build_manual_hybrid_directory(
    osisaf_dir=OSISAF_REPROJECTED_DIR,
    masie_dir=MASIE_REPROJECTED_DIR,
    output_dir=MANUAL_HYBRID_REPROJECTED_DIR,
    threshold=0.15,
    save_previews=True,
):
    """Build manual hybrid matrices for all matching OSISAF and MASIE .npy files."""
    osisaf_dir = Path(osisaf_dir)
    masie_dir = Path(masie_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for osisaf_path in sorted(osisaf_dir.glob("*.npy")):
        date = osisaf_path.stem
        masie_path = masie_dir / osisaf_path.name
        if not masie_path.exists():
            print(f"Skipping {date}: MASIE file not found at {masie_path}")
            continue

        print(date)
        osisaf_matrix = np.load(osisaf_path)
        masie_matrix = np.load(masie_path)
        hybrid_matrix = build_hybrid_matrix(
            osisaf_matrix=osisaf_matrix,
            masie_matrix=masie_matrix,
            threshold=threshold,
        )

        output_path = output_dir / f"{date}.npy"
        np.save(output_path, hybrid_matrix)
        outputs.append(output_path)

        if save_previews:
            save_preview(
                matrix=hybrid_matrix,
                preview_path=output_dir / "preview" / f"{date}.png",
                title=f"{date} - manual hybrid",
            )

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Build manual hybrid baseline arrays.")
    parser.add_argument("--osisaf-dir", type=Path, default=OSISAF_REPROJECTED_DIR)
    parser.add_argument("--masie-dir", type=Path, default=MASIE_REPROJECTED_DIR)
    parser.add_argument("--output-dir", type=Path, default=MANUAL_HYBRID_REPROJECTED_DIR)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()

    build_manual_hybrid_directory(
        osisaf_dir=args.osisaf_dir,
        masie_dir=args.masie_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        save_previews=not args.no_previews,
    )


if __name__ == "__main__":
    main()
