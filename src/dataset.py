import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import Optional, Callable, Tuple, List


class IceConcentrationDataset(Dataset):
    def __init__(
            self,
            osisaf_dir: str,
            masam2_dir: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            date_pattern: str = r'(\d{8})'
    ):
        """
        Инициализация датасета

        Args:
            osisaf_dir: путь к директории с файлами OSISAF (.npy)
            masam2_dir: путь к директории с файлами MASAM2 (.npy)
            transform: преобразования для входных данных (низкое разрешение)
            target_transform: преобразования для таргета (высокое разрешение)
            date_pattern: регулярное выражение для извлечения даты из имени файла
        """
        self.osisaf_dir = Path(osisaf_dir)
        self.masam2_dir = Path(masam2_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.date_pattern = date_pattern
        self.pairs = self._find_matching_pairs()

        if len(self.pairs) == 0:
            raise ValueError(
                f"Не найдено совпадающих пар файлов\n"
            )

        self._init_resolutions()
        print(f"Загружено {len(self.pairs)} пар данных")
        print(f"Разрешение OSISAF (вход): {self.lr_shape}")
        print(f"Разрешение MASAM2 (таргет): {self.hr_shape}")
        print(f"Соотношение разрешений: {self.scale_factor[0]:.2f}x (H) x {self.scale_factor[1]:.2f}x (W)")

    def _find_matching_pairs(self) -> List[Tuple[str, str]]:
        """Поиск пар файлов с одинаковой датой в именах"""
        osisaf_files = {f: self._extract_date(f.name) for f in self.osisaf_dir.glob('*.npy')}
        masam2_files = {f: self._extract_date(f.name) for f in self.masam2_dir.glob('*.npy')}

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
        """Извлечение даты ГГГГММДД из имени файла"""
        match = re.search(self.date_pattern, filename)
        return match.group(1) if match else None

    def _init_resolutions(self):
        """Определение разрешений и соотношения масштабирования"""
        osisaf_sample = np.load(self.pairs[0][0])
        masam2_sample = np.load(self.pairs[0][1])

        self.lr_shape = osisaf_sample.shape
        self.hr_shape = masam2_sample.shape

        self.scale_factor = (
            self.hr_shape[0] / self.lr_shape[0],
            self.hr_shape[1] / self.lr_shape[1]
        )

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Нормализация в диапазон [0, 1] делением на 101.0
        """
        data = data.astype(np.float32)
        data = np.clip(data, 0, 101)
        data = data / 101.0

        return data

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        Получение элемента датасета

        Returns:
            dict с ключами:
                'lr': тензор низкого разрешения [1, H_lr, W_lr]
                'hr': тензор высокого разрешения [1, H_hr, W_hr]
                'date': строка с датой в формате ГГГГММДД
                'lr_path': путь к файлу OSISAF
                'hr_path': путь к файлу MASAM2
        """
        osisaf_path, masam2_path = self.pairs[idx]

        date = self._extract_date(os.path.basename(osisaf_path))

        lr_data = np.load(osisaf_path)
        hr_data = np.load(masam2_path)

        lr_data = self._preprocess_data(lr_data)
        hr_data = self._preprocess_data(hr_data)

        if self.transform:
            lr_data = self.transform(lr_data)

        if self.target_transform:
            hr_data = self.target_transform(hr_data)

        lr_tensor = torch.from_numpy(lr_data).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_data).unsqueeze(0).float()

        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
            'date': date,
            'lr_path': osisaf_path,
            'hr_path': masam2_path
        }

    def get_statistics(self) -> dict:
        """Получение статистики по датасету"""
        stats = {
            'total_pairs': len(self.pairs),
            'lr_resolution': self.lr_shape,
            'hr_resolution': self.hr_shape,
            'scale_factor': self.scale_factor,
            'date_range': (self._extract_date(os.path.basename(self.pairs[0][0])),
                           self._extract_date(os.path.basename(self.pairs[-1][0])))
        }
        return stats


def create_dataloaders(osisaf_dir, masam2_dir, batch_size=16, train_ratio=(0.7, 0.15, 0.15), num_workers=2):
    """
    Создание DataLoader для обучения, валидации и тестирования
    """
    full_dataset = IceConcentrationDataset(
        osisaf_dir=osisaf_dir,
        masam2_dir=masam2_dir
    )

    total_size = len(full_dataset)
    train_size = int(train_ratio[0] * total_size)
    val_size = int(train_ratio[1] * total_size)

    indices = list(range(len(full_dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Размеры датасетов:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader, full_dataset