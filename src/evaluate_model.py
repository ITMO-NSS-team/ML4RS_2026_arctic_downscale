import argparse
import importlib
from pathlib import Path

import numpy as np


MODEL_MODULES = {
    "unet_light": "unet_light",
    "attention_unet": "attention_unet",
    "unet_large": "unet_large",
}


def parse_date_range(value):
    """
    Parse a comma-separated CLI date range.

    Args:
        value: Text in START,END format.

    Returns:
        Tuple with start and end date strings.
    """
    dates = tuple(part.strip() for part in value.split(","))
    if len(dates) != 2:
        raise argparse.ArgumentTypeError("Date range must be START,END")
    return dates


def load_model(model_name, weights_path):
    """
    Load a model architecture and its weights.

    Args:
        model_name: Model key from MODEL_MODULES.
        weights_path: Path to a state dict or checkpoint.

    Returns:
        Model in evaluation mode.
    """
    import torch

    from config import DEVICE

    module = importlib.import_module(MODEL_MODULES[model_name])
    model = module.UNet(in_channels=1, out_channels=1).to(DEVICE)

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def balanced_accuracy(predictions, targets, threshold):
    pred_binary = predictions >= threshold
    target_binary = targets >= threshold

    true_positive = (pred_binary & target_binary).sum().float()
    true_negative = (~pred_binary & ~target_binary).sum().float()
    false_positive = (pred_binary & ~target_binary).sum().float()
    false_negative = (~pred_binary & target_binary).sum().float()

    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive

    scores = []
    if positive_count > 0:
        scores.append(true_positive / positive_count)
    if negative_count > 0:
        scores.append(true_negative / negative_count)

    if not scores:
        return predictions.new_tensor(float("nan"))

    return sum(scores) / len(scores)


def ice_edge_error(predictions, targets, threshold):
    pred_binary = predictions >= threshold
    target_binary = targets >= threshold
    return (pred_binary != target_binary).sum().float() / 1e5


def evaluate_split(model, dataloader, split_name, threshold):
    """
    Evaluate one dataset split with all metrics.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the split.
        split_name: Label for progress output.
        threshold: Ice/no-ice threshold.

    Returns:
        Metric summary dictionary.
    """
    import torch
    import torch.nn.functional as F
    import torchmetrics
    from tqdm import tqdm

    from config import DEVICE

    values = {
        "BACC": [],
        "IIEE": [],
        "MAE": [],
        "PSNR": [],
        "SSIM": [],
    }

    mae_metric = torchmetrics.MeanAbsoluteError().to(DEVICE)
    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(
        DEVICE
    )

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating {split_name}")
        for batch in pbar:
            lr_images = batch["lr"].to(DEVICE)
            hr_images = batch["hr"].to(DEVICE)

            predictions = model(lr_images).clamp(0.0, 1.0)
            if predictions.shape[-2:] != hr_images.shape[-2:]:
                predictions = F.interpolate(
                    predictions,
                    size=hr_images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            for prediction, target in zip(predictions, hr_images):
                prediction = prediction.unsqueeze(0)
                target = target.unsqueeze(0)

                values["BACC"].append(
                    balanced_accuracy(prediction, target, threshold).item()
                )
                values["IIEE"].append(
                    ice_edge_error(prediction, target, threshold).item()
                )
                values["MAE"].append(mae_metric(prediction, target).item())
                values["PSNR"].append(psnr_metric(prediction, target).item())
                values["SSIM"].append(ssim_metric(prediction, target).item())

            pbar.set_postfix(
                {
                    "MAE": f"{np.nanmean(values['MAE']):.4f}",
                    "PSNR": f"{np.nanmean(values['PSNR']):.2f}",
                }
            )

    return {
        metric: {
            "mean": float(np.nanmean(metric_values)),
            "std": float(np.nanstd(metric_values, ddof=1))
            if len(metric_values) > 1
            else 0.0,
        }
        for metric, metric_values in values.items()
    }


def format_mean_std(metric_values, precision=3):
    """Format metric mean and standard deviation"""
    return (
        f"{metric_values['mean']:.{precision}f} +/- "
        f"{metric_values['std']:.{precision}f}"
    )


def print_results_table(model_label, results):
    """
    Print evaluation metrics as a comparison table.

    Args:
        model_label: Label for the evaluated model.
        results: Metrics grouped by split.
    """
    print("\n" + "=" * 96)
    print(f"Performance comparison for {model_label}")
    print("=" * 96)
    print(
        f"{'Method':<16}{'Split':<8}{'BACC':<18}{'IIEE (x10^5)':<18}"
        f"{'MAE':<18}{'PSNR':<18}{'SSIM':<18}"
    )
    print("-" * 96)

    for idx, split_name in enumerate(("Train", "Val", "Test")):
        method = model_label if idx == 0 else ""
        split_results = results[split_name.lower()]
        print(
            f"{method:<16}{split_name:<8}"
            f"{format_mean_std(split_results['BACC']):<18}"
            f"{format_mean_std(split_results['IIEE']):<18}"
            f"{format_mean_std(split_results['MAE']):<18}"
            f"{format_mean_std(split_results['PSNR'], precision=2):<18}"
            f"{format_mean_std(split_results['SSIM']):<18}"
        )


def evaluate_model(
    model_name,
    weights_path,
    train_date_range,
    val_date_range,
    test_date_range,
    batch_size=8,
    num_workers=2,
    threshold=0.15,
    with_missed=False,
):
    """
    Evaluate a trained model on train, validation, and test splits.

    Args:
        model_name: Model key from MODEL_MODULES.
        weights_path: Path to weights or checkpoint.
        train_date_range: Training range as start and end dates.
        val_date_range: Validation range as start and end dates.
        test_date_range: Test range as start and end dates.
        batch_size: Evaluation batch size.
        num_workers: Number of DataLoader workers.
        threshold: Ice/no-ice threshold.
        with_missed: Whether to keep MASAM2 missed-date pairs.

    Returns:
        Metrics grouped by split.
    """
    from config import MASAM2_DIR, OSISAF_DIR
    from dataset import create_dataloaders

    model = load_model(model_name, weights_path)
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        osisaf_dir=OSISAF_DIR,
        masam2_dir=MASAM2_DIR,
        train_date_range=train_date_range,
        val_date_range=val_date_range,
        test_date_range=test_date_range,
        batch_size=batch_size,
        num_workers=num_workers,
        with_missed=with_missed,
    )

    results = {
        "train": evaluate_split(model, train_loader, "train", threshold),
        "val": evaluate_split(model, val_loader, "validation", threshold),
        "test": evaluate_split(model, test_loader, "test", threshold),
    }

    print_results_table(model_name, results)
    return results


def main():
    """Parse CLI arguments and run model evaluation"""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained U-Net model on train/val/test date splits."
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_MODULES.keys()),
        default="unet_light",
        help="Model architecture used for the checkpoint.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to .pth weights or training checkpoint.",
    )
    parser.add_argument(
        "--train-range",
        type=parse_date_range,
        default=("20120701", "20201231"),
        help="Train date range as START,END in YYYYMMDD format.",
    )
    parser.add_argument(
        "--val-range",
        type=parse_date_range,
        default=("20210101", "20221231"),
        help="Validation date range as START,END in YYYYMMDD format.",
    )
    parser.add_argument(
        "--test-range",
        type=parse_date_range,
        default=("20230101", "20250630"),
        help="Test date range as START,END in YYYYMMDD format.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Ice/no-ice threshold for BACC and IIEE.",
    )
    parser.add_argument(
        "--with-missed",
        action="store_true",
        help="Include MASAM2 dates listed in masam2_missed.txt.",
    )

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        weights_path=args.weights,
        train_date_range=args.train_range,
        val_date_range=args.val_range,
        test_date_range=args.test_range,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        with_missed=args.with_missed,
    )


if __name__ == "__main__":
    main()
