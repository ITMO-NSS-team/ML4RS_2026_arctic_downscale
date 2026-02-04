import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import IceConcentrationDataset
from tqdm import tqdm
import time
import torchmetrics

from config import OSISAF_DIR, MASAM2_DIR, DEVICE

def evaluate_interpolation_methods(
        osisaf_dir,
        masam2_dir,
        batch_size=16,
):
    """
    Оценивает различные методы интерполяции для повышения разрешения
    """
    print("=" * 70)
    print("ОЦЕНКА МЕТОДОВ ИНТЕРПОЛЯЦИИ ДЛЯ ПОВЫШЕНИЯ РАЗРЕШЕНИЯ")
    print("=" * 70)

    dataset = IceConcentrationDataset(osisaf_dir=osisaf_dir, masam2_dir=masam2_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    metrics_dict = {}
    methods = ('bilinear', 'bicubic', 'nearest')

    for method in methods:
        print(f"\nОценка метода: {method.upper()}")

        mae_metric = torchmetrics.MeanAbsoluteError().to(DEVICE)
        mse_metric = torchmetrics.MeanSquaredError().to(DEVICE)
        psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

        pbar = tqdm(dataloader, desc=f"Обработка {method}")

        for batch in pbar:
            lr_images = batch['lr'].to(DEVICE)
            hr_images = batch['hr'].to(DEVICE)

            with torch.no_grad():
                target_size = hr_images.shape[-2:]

                if method == 'bilinear':
                    upsampled = F.interpolate(
                        lr_images,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )
                elif method == 'bicubic':
                    upsampled = F.interpolate(
                        lr_images,
                        size=target_size,
                        mode='bicubic',
                        align_corners=False
                    )
                elif method == 'nearest':
                    upsampled = F.interpolate(
                        lr_images,
                        size=target_size,
                        mode='nearest'
                    )

                mae_metric.update(upsampled, hr_images)
                mse_metric.update(upsampled, hr_images)
                psnr_metric.update(upsampled, hr_images)
                ssim_metric.update(upsampled, hr_images)

                pbar.set_postfix({
                    'MAE': f'{mae_metric.compute():.4f}',
                    'PSNR': f'{psnr_metric.compute():.2f}'
                })

        mae = mae_metric.compute().item()
        mse = mse_metric.compute().item()
        rmse = np.sqrt(mse)
        psnr = psnr_metric.compute().item()
        ssim = ssim_metric.compute().item()

        metrics_dict[method] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'PSNR': psnr,
            'SSIM': ssim
        }

        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")

    return metrics_dict




def main():
    methods_to_test = ['bilinear', 'bicubic', 'nearest']
    print(f"Тестируемые методы: {methods_to_test}")

    start_time = time.time()

    metrics_dict = evaluate_interpolation_methods(
        osisaf_dir=OSISAF_DIR,
        masam2_dir=MASAM2_DIR,
        batch_size=8,
    )

    elapsed_time = time.time() - start_time
    print(f"\nОбщее время оценки: {elapsed_time:.1f} секунд")

    best_mae_method = min(metrics_dict.keys(), key=lambda x: metrics_dict[x]['MAE'])
    best_psnr_method = max(metrics_dict.keys(), key=lambda x: metrics_dict[x]['PSNR'])
    best_ssim_method = max(metrics_dict.keys(), key=lambda x: metrics_dict[x]['SSIM'])

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 70)
    print(f"Лучший по MAE:  {best_mae_method}  ({metrics_dict[best_mae_method]['MAE']:.6f})")
    print(f"Лучший по PSNR: {best_psnr_method}  ({metrics_dict[best_psnr_method]['PSNR']:.2f} dB)")
    print(f"Лучший по SSIM: {best_ssim_method}  ({metrics_dict[best_ssim_method]['SSIM']:.4f})")

    return metrics_dict


if __name__ == "__main__":
    metrics = main()