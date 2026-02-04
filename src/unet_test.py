import torch
import numpy as np
import torchmetrics
from tqdm import tqdm
import os
import random

from dataset import create_dataloaders
from unet import UNet
from config import DEVICE, OSISAF_DIR, MASAM2_DIR
from visualization import visualize_results


def load_model():
    """Загрузка обученной модели"""
    model = UNet().to(DEVICE)
    if os.path.exists('../models/unet_final_model.pth'):
        checkpoint = torch.load('../models/unet_final_model.pth', map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(f"модель загружена из {'../models/unet_final_model.pth'}")
    else:
        print(f"Внимание: файл {'../models/unet_final_model.pth'} не найден!")
        model = None

    return model


def create_test_dataset():
    """
    Создание тестового датасета
    """
    print(f"Создание тестового набора...")

    train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
        osisaf_dir=OSISAF_DIR,
        masam2_dir=MASAM2_DIR,
        batch_size=8,
        train_ratio=(0.7, 0.15, 0.15)
    )

    return test_loader


def test_model_on_dataset(model, test_loader, device='cuda', batch_size=4):
    """
    Тестирование модели на тестовом наборе

    """
    print(f"Тестирование модели (batch_size={batch_size})...")

    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    mse_metric = torchmetrics.MeanSquaredError().to(device)
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    pbar = tqdm(test_loader, desc="Тестирование", unit="batch")

    with torch.no_grad():
        for batch in pbar:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            pred = model(lr)

            mae_metric.update(pred, hr)
            mse_metric.update(pred, hr)
            psnr_metric.update(pred, hr)
            ssim_metric.update(pred, hr)

            current_mae = mae_metric.compute().item()
            current_psnr = psnr_metric.compute().item()
            pbar.set_postfix({
                'MAE': f'{current_mae:.4f}',
                'PSNR': f'{current_psnr:.2f}'
            })

        mae = mae_metric.compute().item()
        mse = mse_metric.compute().item()
        rmse = np.sqrt(mse)
        psnr = psnr_metric.compute().item()
        ssim = ssim_metric.compute().item()

        metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'PSNR': psnr, 'SSIM': ssim, }

        print(f"\nРезультаты тестирования:")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  PSNR: {metrics['PSNR']:.2f} dB")
        print(f"  SSIM: {metrics['SSIM']:.4f}")

        return metrics


def visualize_random_pair(model, test_loader, device='cuda'):
    """
    Берет случайную пару из тестового датасета и визуализирует
    """
    batch = next(iter(test_loader))

    idx = random.randint(0, len(batch['lr']) - 1)

    lr = batch['lr'][idx].unsqueeze(0).to(device)  # Добавляем dimension для батча
    hr = batch['hr'][idx].unsqueeze(0).to(device)
    date = batch['date'][idx]

    with torch.no_grad():
        model.eval()
        pred = model(lr)

    lr_cpu = lr.cpu()
    hr_cpu = hr.cpu()
    pred_cpu = pred.cpu()

    save_path = visualize_results(
        lr_images=lr_cpu,
        hr_images=hr_cpu,
        pred_images=pred_cpu,
        epoch='None',
        model_name=f"U-Net | Date: {date}",
        save_path=f"random_example_{date}.png"
    )

    print(f"Визуализация сохранена: {save_path}")

    return lr_cpu, hr_cpu, pred_cpu, date


def run_full_test(batch_size=4):
    print("=" * 70)
    print("U-NET TESTING")
    print("=" * 70)

    model= load_model()

    test_loader = create_test_dataset()

    metrics = test_model_on_dataset(model, test_loader, DEVICE, batch_size)

    visualize_random_pair(model, test_loader, DEVICE)

    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    if not os.path.exists(OSISAF_DIR) or not os.path.exists(MASAM2_DIR):
        print(" Директории с данными не найдены")
    else:
        try:
            metrics = run_full_test(batch_size=8)
        except Exception as e:
            print(f"Ошибка: {e}")