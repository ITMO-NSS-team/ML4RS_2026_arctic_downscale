import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from tqdm import tqdm

from config import DEVICE, MASAM2_DIR, OSISAF_DIR, OUTPUT_FIGURES_DIR, OUTPUT_MODELS_DIR
from dataset import create_dataloaders
from models.attention_unet import AttentionUNet
from visualization.visualization import plot_metrics, visualize_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = OUTPUT_MODELS_DIR
BEST_MODEL_PATH = MODELS_DIR / "attention_unet_best.pth"
FINAL_MODEL_PATH = MODELS_DIR / "attention_unet_final.pth"


def train_attention_unet(
    model=None,
    train_loader=None,
    val_loader=None,
    test_loader=None,
    epochs=100,
    learning_rate=0.001,
    visualize_every=5,
    plot_every=10,
    patience=10,
):
    """
    Train the attention U-Net and save the best checkpoint.

    Args:
        model: Optional model instance to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: Optional DataLoader for visualization samples.
        epochs: Maximum number of training epochs.
        learning_rate: Adam optimizer learning rate.
        visualize_every: Epoch interval for result visualizations.
        plot_every: Epoch interval for metric plots.
        patience: Early-stopping patience in epochs.

    Returns:
        Dictionary with the model and metric histories.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRAINING ATTENTION U-NET MODEL")
    print("=" * 70)

    if model is None:
        model = AttentionUNet(in_channels=1, out_channels=1).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.L1Loss()

    train_rmse = torchmetrics.MeanSquaredError(squared=False).to(DEVICE)
    train_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    train_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(
        DEVICE
    )

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

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nTraining started...")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {DEVICE}")

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        epoch_train_loss = 0

        train_rmse.reset()
        train_psnr.reset()
        train_ssim.reset()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            lr_images = batch["lr"].to(DEVICE)
            hr_images = batch["hr"].to(DEVICE)

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

            pbar.set_postfix(
                {
                    "train_loss": f"{current_loss:.4f}",
                    "avg_train": f"{epoch_train_loss / (len(pbar) + 1):.4f}",
                }
            )

        train_rmse_value = train_rmse.compute().item()
        train_psnr_value = train_psnr.compute().item()

        model.eval()
        epoch_val_loss = 0

        val_rmse.reset()
        val_psnr.reset()
        val_ssim.reset()

        with torch.no_grad():
            for batch in val_loader:
                lr_images = batch["lr"].to(DEVICE)
                hr_images = batch["hr"].to(DEVICE)

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

        print(f"\nEpoch {epoch + 1} completed:")
        print(f"Train loss: {avg_epoch_train_loss:.6f}")
        print(f"Val loss:  {avg_epoch_val_loss:.6f}")
        print(
            f"Train RMSE: {train_rmse_value:.4f}, Train PSNR: {train_psnr_value:.2f} dB"
        )
        print(f"Val RMSE:   {val_rmse_value:.4f}, Val PSNR:   {val_psnr_value:.2f} dB")

        if (epoch + 1) % 5 == 0:
            train_ssim_value = train_ssim.compute().item()
            val_ssim_value = val_ssim.compute().item()
            train_ssim_history.append(train_ssim_value)
            val_ssim_history.append(val_ssim_value)
            print(
                f"Train SSIM: {train_ssim_value:.4f}, Val SSIM:   {val_ssim_value:.4f}"
            )

        print(f"Time: {epoch_time:.1f} sec")

        if (epoch + 1) % visualize_every == 0 and test_loader is not None:
            print(f"\nVisualization of epoch results{epoch + 1}...")

            test_batch = next(iter(test_loader))
            lr_test = test_batch["lr"][:4].to(DEVICE)
            hr_test = test_batch["hr"][:4].to(DEVICE)
            dates_test = test_batch["date"][:4]

            with torch.no_grad():
                pred_test = model(lr_test)

            visualize_results(
                lr_images=lr_test.cpu(),
                hr_images=hr_test.cpu(),
                pred_images=pred_test.cpu(),
                epoch=epoch + 1,
                model_name="Attention U-Net Super Resolution",
                save_path=OUTPUT_FIGURES_DIR / f"attention_unet_results_epoch_{epoch + 1}.png",
                dates=dates_test,
            )

        if (epoch + 1) % plot_every == 0:
            print(f"\nGraphs update...")

            plot_metrics(
                epoch_train_losses,
                epoch_val_losses,
                title=f"U-Net - Loss (epoch {epoch + 1})",
                metric="Loss (MAE)",
                save_path=OUTPUT_FIGURES_DIR / f"attention_unet_loss_epoch_{epoch + 1}.png",
            )

            plot_metrics(
                train_rmse_history,
                val_rmse_history,
                title=f"U-Net - RMSE (epoch {epoch + 1})",
                metric="RMSE",
                save_path=OUTPUT_FIGURES_DIR / f"attention_unet_rmse_epoch_{epoch + 1}.png",
            )

            plot_metrics(
                train_psnr_history,
                val_psnr_history,
                title=f"U-Net - PSNR (epoch {epoch + 1})",
                metric="PSNR",
                save_path=OUTPUT_FIGURES_DIR / f"attention_unet_psnr_epoch_{epoch + 1}.png",
            )

            if len(train_ssim_history) > 0:
                plot_metrics(
                    train_ssim_history,
                    val_ssim_history,
                    title=f"U-Net - SSIM (epoch {epoch + 1}, every 5 epoch)",
                    metric="SSIM",
                    save_path=OUTPUT_FIGURES_DIR / f"attention_unet_ssim_epoch_{epoch + 1}.png",
                )

        print("-" * 50)

        if avg_epoch_val_loss < best_val_loss:
            best_val_loss = avg_epoch_val_loss
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_epoch_train_loss,
                    "val_loss": avg_epoch_val_loss,
                    "train_rmse": train_rmse_value,
                    "val_rmse": val_rmse_value,
                },
                BEST_MODEL_PATH,
            )

            print(f"Saved best model (val loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print(f"\n" + "=" * 70)
                print(f"EARLY STOP")
                print(f"Val loss did not improve {patience} epoch in a row")
                print(f"Best val loss: {best_val_loss:.6f}")
                print("=" * 70)
                break

    return {
        "model": model,
        "train_losses": train_losses,
        "epoch_train_losses": epoch_train_losses,
        "epoch_val_losses": epoch_val_losses,
        "train_rmse_history": train_rmse_history,
        "train_psnr_history": train_psnr_history,
        "train_ssim_history": train_ssim_history,
        "val_rmse_history": val_rmse_history,
        "val_psnr_history": val_psnr_history,
        "val_ssim_history": val_ssim_history,
    }


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("GPU not found, using CPU")

    train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
        osisaf_dir=OSISAF_DIR,
        masam2_dir=MASAM2_DIR,
        train_date_range=("20120701", "20201231"),
        val_date_range=("20210101", "20221231"),
        test_date_range=("20230101", "20250630"),
        batch_size=8,
        with_missed=False,
    )

    model = AttentionUNet(in_channels=1, out_channels=1).to(DEVICE)

    results = train_attention_unet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=100,
        learning_rate=0.001,
        visualize_every=5,
        plot_every=5,
        patience=10,
    )

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model saved as: '{FINAL_MODEL_PATH}'")
