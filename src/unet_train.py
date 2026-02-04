import torch
import torch.nn as nn
import time
from tqdm import tqdm
import torch.optim as optim
import torchmetrics

from dataset import create_dataloaders
from unet import UNet
from src.visualization import plot_metrics, visualize_results
from config import DEVICE, OSISAF_DIR, MASAM2_DIR




def train_unet(model=None, train_loader=None, val_loader=None, test_loader=None, epochs=100, learning_rate=0.001, visualize_every=5, plot_every=10, patience=10):
    """
    Обучение U-Net
    """
    print("=" * 70)
    print("ОБУЧЕНИЕ U-NET МОДЕЛИ")
    print("=" * 70)

    if model is None:
        model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.L1Loss()

    train_rmse = torchmetrics.MeanSquaredError(squared=False).to(DEVICE)
    train_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    train_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    val_rmse = torchmetrics.MeanSquaredError(squared=False).to(DEVICE)
    val_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    val_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    train_losses = []
    epoch_train_losses = []
    epoch_val_losses = []

    train_rmse_history = []
    train_psnr_history = []
    train_ssim_history = []
    val_rmse_history = []
    val_psnr_history = []
    val_ssim_history = []

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nНачинаем обучение...")
    print(f"Количество эпох: {epochs}")
    print(f"Размер батча: {train_loader.batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Устройство: {DEVICE}")

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        epoch_train_loss = 0

        train_rmse.reset()
        train_psnr.reset()
        train_ssim.reset()

        pbar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{epochs}")

        for batch in pbar:
            lr_images = batch['lr'].to(DEVICE)
            hr_images = batch['hr'].to(DEVICE)

            optimizer.zero_grad()

            predictions = model(lr_images)

            loss = criterion(predictions, hr_images)

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_train_loss += current_loss
            train_losses.append(current_loss)

            with torch.no_grad():
                train_rmse.update(predictions, hr_images)
                train_psnr.update(predictions, hr_images)

                if (epoch + 1) % 5 == 0:
                    train_ssim.update(predictions, hr_images)

            pbar.set_postfix({
                'train_loss': f'{current_loss:.4f}',
                'avg_train': f'{epoch_train_loss / (len(pbar) + 1):.4f}'
            })

        train_rmse_value = train_rmse.compute().item()
        train_psnr_value = train_psnr.compute().item()

        model.eval()
        epoch_val_loss = 0

        val_rmse.reset()
        val_psnr.reset()
        val_ssim.reset()

        with torch.no_grad():
            for batch in val_loader:
                lr_images = batch['lr'].to(DEVICE)
                hr_images = batch['hr'].to(DEVICE)

                predictions = model(lr_images)
                loss = criterion(predictions, hr_images)
                epoch_val_loss += loss.item()

                val_rmse.update(predictions, hr_images)
                val_psnr.update(predictions, hr_images)

                if (epoch + 1) % 5 == 0:
                    val_ssim.update(predictions, hr_images)

        val_rmse_value = val_rmse.compute().item()
        val_psnr_value = val_psnr.compute().item()

        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        avg_epoch_val_loss = epoch_val_loss / len(val_loader)

        epoch_train_losses.append(avg_epoch_train_loss)
        epoch_val_losses.append(avg_epoch_val_loss)

        scheduler.step(avg_epoch_val_loss)

        epoch_time = time.time() - epoch_start

        train_rmse_history.append(train_rmse_value)
        train_psnr_history.append(train_psnr_value)
        val_rmse_history.append(val_rmse_value)
        val_psnr_history.append(val_psnr_value)

        print(f"\nЭпоха {epoch + 1} завершена:")
        print(f"Train loss: {avg_epoch_train_loss:.6f}")
        print(f"Val loss:  {avg_epoch_val_loss:.6f}")
        print(f"Train RMSE: {train_rmse_value:.4f}, Train PSNR: {train_psnr_value:.2f} dB")
        print(f"Val RMSE:   {val_rmse_value:.4f}, Val PSNR:   {val_psnr_value:.2f} dB")

        if (epoch + 1) % 5 == 0:
            train_ssim_value = train_ssim.compute().item()
            val_ssim_value = val_ssim.compute().item()
            train_ssim_history.append(train_ssim_value)
            val_ssim_history.append(val_ssim_value)
            print(f"Train SSIM: {train_ssim_value:.4f}, Val SSIM:   {val_ssim_value:.4f}")

        print(f"Время: {epoch_time:.1f} сек")

        if (epoch + 1) % visualize_every == 0 and test_loader is not None:
            print(f"\nВизуализация результатов эпохи {epoch + 1}...")

            test_batch = next(iter(test_loader))
            lr_test = test_batch['lr'][:4].to(DEVICE)
            hr_test = test_batch['hr'][:4].to(DEVICE)

            with torch.no_grad():
                pred_test = model(lr_test)

            visualize_results(
                lr_images=lr_test.cpu(),
                hr_images=hr_test.cpu(),
                pred_images=pred_test.cpu(),
                epoch=epoch + 1,
                model_name="U-Net Super Resolution"
            )

        if (epoch + 1) % plot_every == 0:
            print(f"\nОбновление графиков...")

            plot_metrics(epoch_train_losses, epoch_val_losses, title=f"U-Net - Потери (эпоха {epoch + 1})", metric="Loss (MAE)")

            plot_metrics(train_rmse_history, val_rmse_history, title=f"U-Net - RMSE (эпоха {epoch + 1})", metric="RMSE")

            plot_metrics(train_psnr_history, val_psnr_history, title=f"U-Net - PSNR (эпоха {epoch + 1})", metric="PSNR")

            if len(train_ssim_history) > 0:
                plot_metrics(train_ssim_history, val_ssim_history, title=f"U-Net - SSIM (эпоха {epoch + 1}, каждые 5 эпох)", metric="SSIM")

        print("-" * 50)

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_train_loss,
                'val_loss': avg_epoch_val_loss,
                'train_rmse': train_rmse_value,
                'val_rmse': val_rmse_value,
            }, '../models/unet_best_model.pth')

            print(f"Сохранена лучшая модель (val loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"\n" + "=" * 70)
                print(f"РАННЯЯ ОСТАНОВКА")
                print(f"Val loss не улучшался {patience} эпох подряд")
                print(f"Лучший val loss: {best_val_loss:.6f}")
                print("=" * 70)
                break


    return {
        'model': model,
        'train_losses': train_losses,
        'epoch_train_losses': epoch_train_losses,
        'epoch_val_losses': epoch_val_losses,
        'train_rmse_history': train_rmse_history,
        'train_psnr_history': train_psnr_history,
        'train_ssim_history': train_ssim_history,
        'val_rmse_history': val_rmse_history,
        'val_psnr_history': val_psnr_history,
        'val_ssim_history': val_ssim_history,
    }



if __name__ == "__main__":

    print(f"Используемое устройство: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("GPU не найден, используем CPU")

    train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
        osisaf_dir=OSISAF_DIR,
        masam2_dir=MASAM2_DIR,
        batch_size=8,
        train_ratio=(0.7, 0.15, 0.15)
    )

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    results = train_unet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=100,
        learning_rate=0.001,
        visualize_every=5,
        plot_every=5,
        patience=10
    )

    checkpoint = torch.load('../models/unet_best_model.pth', map_location=DEVICE)
    model_state_dict = checkpoint['../models/model_state_dict']
    model.load_state_dict(model_state_dict)
    torch.save(results['model'].state_dict(), '../models/unet_final_model.pth')
    print("Финальная модель сохранена как '../models/unet_final_model.pth'")
