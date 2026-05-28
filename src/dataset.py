import ast
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class IceConcentrationDataset(Dataset):
    def __init__(
        self,
        osisaf_dir: str,
        masam2_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        date_pattern: str = r"(\d{8})",
    ):
        """
        Initializing the dataset

        Args:
            osisaf_dir: Path to directory with OSISAF (.npy)
            masam2_dir: Path to directory with  MASAM2 (.npy)
            transform: transformations for input data (low resolution)
            target_transform: conversions for target (high resolution)
            date_pattern: regular expression for extracting a date from a file name
        """
        self.osisaf_dir = Path(osisaf_dir)
        self.masam2_dir = Path(masam2_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.date_pattern = date_pattern
        self.pairs = self._find_matching_pairs()

        if len(self.pairs) == 0:
            raise ValueError(f"No matching file pairs found\n")

        self._init_resolutions()
        print(f"Uploaded {len(self.pairs)} pairs")
        print(f"Resolution OSISAF (input): {self.lr_shape}")
        print(f"Resolution MASAM2 (target): {self.hr_shape}")
        print(
            f"Resolution ratio: {self.scale_factor[0]:.2f}x (H) x {self.scale_factor[1]:.2f}x (W)"
        )

    def _find_matching_pairs(self) -> List[Tuple[str, str]]:
        osisaf_files = {
            f: self._extract_date(f.name) for f in self.osisaf_dir.rglob("*.npy")
        }
        masam2_files = {
            f: self._extract_date(f.name) for f in self.masam2_dir.rglob("*.npy")
        }

        osisaf_files = {f: d for f, d in osisaf_files.items() if d is not None}
        masam2_files = {f: d for f, d in masam2_files.items() if d is not None}

        osisaf_by_date = {}
        for f, date in osisaf_files.items():
            osisaf_by_date.setdefault(date, []).append(f)

        masam2_by_date = {}
        for f, date in masam2_files.items():
            masam2_by_date.setdefault(date, []).append(f)

        pairs = []
        for date in set(osisaf_by_date.keys()) & set(masam2_by_date.keys()):
            osisaf_path = osisaf_by_date[date][0]
            masam2_path = masam2_by_date[date][0]
            pairs.append((str(osisaf_path), str(masam2_path)))

        pairs.sort(key=lambda x: x[0])
        return pairs

    def _extract_date(self, filename: str) -> Optional[str]:
        match = re.search(self.date_pattern, filename)
        return match.group(1) if match else None

    def _init_resolutions(self):
        osisaf_sample = np.load(self.pairs[0][0])
        masam2_sample = np.load(self.pairs[0][1])

        self.lr_shape = osisaf_sample.shape
        self.hr_shape = masam2_sample.shape

        self.scale_factor = (
            self.hr_shape[0] / self.lr_shape[0],
            self.hr_shape[1] / self.lr_shape[1],
        )

    def _preprocess_osisaf(self, data: np.ndarray) -> np.ndarray:
        """Normalize OSISAF concentrations stored as percentages to [0, 1]."""
        data = np.clip(data, 0, 100).astype(np.float32)
        return data / 100.0

    def _preprocess_masam2(self, data: np.ndarray) -> np.ndarray:
        """Convert already normalized MASAM2 target data to model precision."""
        return data.astype(np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        Getting a dataset element

        Returns:
            dict with keys:
                'lr': low resolution tensor [1, H_lr, W_lr]
                'hr': high resolution tensor [1, H_hr, W_hr]
                'date': string with data in format YYYYMMDD
                'lr_path': path to OSISAF file
                'hr_path': path to  MASAM2 file
        """
        osisaf_path, masam2_path = self.pairs[idx]

        date = self._extract_date(os.path.basename(osisaf_path))

        lr_data = np.load(osisaf_path)
        hr_data = np.load(masam2_path)

        lr_data = self._preprocess_osisaf(lr_data)
        hr_data = self._preprocess_masam2(hr_data)

        if self.transform:
            lr_data = self.transform(lr_data)

        if self.target_transform:
            hr_data = self.target_transform(hr_data)

        lr_tensor = torch.from_numpy(lr_data).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_data).unsqueeze(0).float()

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "date": date,
            "lr_path": osisaf_path,
            "hr_path": masam2_path,
        }

    def get_statistics(self) -> dict:
        stats = {
            "total_pairs": len(self.pairs),
            "lr_resolution": self.lr_shape,
            "hr_resolution": self.hr_shape,
            "scale_factor": self.scale_factor,
            "date_range": (
                self._extract_date(os.path.basename(self.pairs[0][0])),
                self._extract_date(os.path.basename(self.pairs[-1][0])),
            ),
        }
        return stats


def _validate_date_range(name: str, date_range: Tuple[str, str]) -> Tuple[str, str]:
    if len(date_range) != 2:
        raise ValueError(f"{name} date range must contain a start and an end date")

    start, end = date_range
    for value in (start, end):
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as error:
            raise ValueError(
                f"{name} date range value '{value}' must use YYYYMMDD format"
            ) from error

    if start > end:
        raise ValueError(f"{name} date range start must not be later than its end")

    return start, end


def _load_missed_dates(missed_path: Path) -> set[str]:
    if not missed_path.exists():
        raise FileNotFoundError(
            f"MASAM2 missed-dates file not found: {missed_path}"
        )

    try:
        entries = ast.literal_eval(missed_path.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"Unable to parse MASAM2 missed-dates file: {missed_path}"
        ) from error

    if not isinstance(entries, (list, tuple, set)):
        raise ValueError("MASAM2 missed-dates file must contain a sequence of filenames")

    missed_dates = set()
    for entry in entries:
        match = re.search(r"(\d{8})", str(entry))
        if match is None:
            raise ValueError(f"Unable to extract a date from missed entry: {entry}")
        missed_dates.add(match.group(1))

    return missed_dates


def create_dataloaders(
    osisaf_dir,
    masam2_dir,
    train_date_range: Tuple[str, str],
    val_date_range: Tuple[str, str],
    test_date_range: Tuple[str, str],
    batch_size=16,
    num_workers=2,
    with_missed: bool = False,
):
    """
    Create data loaders from sequential, inclusive date ranges.

    Ranges must use YYYYMMDD format and be ordered as train, validation, test.
    Set with_missed=False to exclude dates listed in masam2_missed.txt located
    next to the OSISAF and MASAM2 directories.
    """
    full_dataset = IceConcentrationDataset(osisaf_dir=osisaf_dir, masam2_dir=masam2_dir)

    train_range = _validate_date_range("train", train_date_range)
    val_range = _validate_date_range("validation", val_date_range)
    test_range = _validate_date_range("test", test_date_range)

    if not (train_range[1] < val_range[0] and val_range[1] < test_range[0]):
        raise ValueError(
            "Date ranges must be non-overlapping and ordered as train, validation, test"
        )

    train_indices = []
    val_indices = []
    test_indices = []
    missed_dates = set()
    if not with_missed:
        missed_path = Path(masam2_dir).parent / "masam2_missed.txt"
        missed_dates = _load_missed_dates(missed_path)
    excluded_pairs = 0

    for idx, (osisaf_path, _) in enumerate(full_dataset.pairs):
        date = full_dataset._extract_date(os.path.basename(osisaf_path))

        if date in missed_dates:
            excluded_pairs += 1
            continue

        if train_range[0] <= date <= train_range[1]:
            train_indices.append(idx)
        elif val_range[0] <= date <= val_range[1]:
            val_indices.append(idx)
        elif test_range[0] <= date <= test_range[1]:
            test_indices.append(idx)

    for name, indices in (
        ("train", train_indices),
        ("validation", val_indices),
        ("test", test_indices),
    ):
        if not indices:
            raise ValueError(f"{name} date range has no matching data pairs")

    print("Split method: sequential date ranges")
    print(f"  Train: {train_range[0]} - {train_range[1]}")
    print(f"  Validation: {val_range[0]} - {val_range[1]}")
    print(f"  Test: {test_range[0]} - {test_range[1]}")
    if not with_missed:
        print(f"Excluded MASAM2 missed pairs: {excluded_pairs}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Datasets sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, full_dataset
