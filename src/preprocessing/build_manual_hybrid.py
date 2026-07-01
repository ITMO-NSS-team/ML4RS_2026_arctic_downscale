import argparse
import re
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

    masie_by_date = index_npy_files_by_date(masie_dir)
    outputs = []
    for osisaf_path in sorted(osisaf_dir.rglob("*.npy")):
        date = extract_date_from_filename(osisaf_path.name)
        if date is None:
            print(f"Skipping {osisaf_path}: no YYYYMMDD date in filename")
            continue

        masie_path = masie_by_date.get(date)
        if masie_path is None:
            print(f"Skipping {date}: MASIE file not found under {masie_dir}")
            continue

        print(date)
        osisaf_matrix = np.load(osisaf_path)
        masie_matrix = np.load(masie_path)
        hybrid_matrix = build_hybrid_matrix(
            osisaf_matrix=osisaf_matrix,
            masie_matrix=masie_matrix,
            threshold=threshold,
        )

        year_dir = output_dir / date[:4]
        year_dir.mkdir(parents=True, exist_ok=True)
        output_path = year_dir / f"{date}.npy"
        np.save(output_path, hybrid_matrix)
        outputs.append(output_path)

        if save_previews:
            save_preview(
                matrix=hybrid_matrix,
                preview_path=year_dir / "preview" / f"{date}.png",
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
