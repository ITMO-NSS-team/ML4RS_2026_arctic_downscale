import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


INTERPOLATION_METHODS = ("bilinear", "bicubic", "nearest")


def parse_date_range(value):
    """Parse a comma-separated CLI date range"""
    dates = tuple(part.strip() for part in value.split(","))
    if len(dates) != 2:
        raise argparse.ArgumentTypeError("Date range must be START,END")
    return dates


def parse_methods(value):
    """Parse and validate interpolation methods from CLI text"""
    methods = tuple(method.strip().lower() for method in value.split(","))
    unknown_methods = sorted(set(methods) - set(INTERPOLATION_METHODS))
    if unknown_methods:
        raise argparse.ArgumentTypeError(
            f"Unknown interpolation methods: {', '.join(unknown_methods)}"
        )
    return methods


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


def interpolate_images(lr_images, target_size, method):
    """
    Upsample low-resolution images to a target size.

    Args:
        lr_images: Input image batch.
        target_size: Target height and width.
        method: Interpolation method.

    Returns:
        Upsampled image batch.
    """
    import torch.nn.functional as F

    if method == "nearest":
        return F.interpolate(lr_images, size=target_size, mode=method)

    return F.interpolate(
        lr_images,
        size=target_size,
        mode=method,
        align_corners=False,
    )


def summarize_metric_values(values):
    """Summarize metric samples with mean and standard deviation"""
    return {
        metric: {
            "mean": float(np.nanmean(metric_values)),
            "std": float(np.nanstd(metric_values, ddof=1))
            if len(metric_values) > 1
            else 0.0,
        }
        for metric, metric_values in values.items()
    }


def evaluate_interpolation_split(method, dataloader, split_name, threshold):
    """
    Evaluate one interpolation method on one data split.

    Args:
        method: Interpolation method name.
        dataloader: DataLoader for the split.
        split_name: Label for progress output.
        threshold: Ice/no-ice threshold.

    Returns:
        Metric summary dictionary.
    """
    import torch
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
        pbar = tqdm(dataloader, desc=f"{method} on {split_name}")
        for batch in pbar:
            lr_images = batch["lr"].to(DEVICE)
            hr_images = batch["hr"].to(DEVICE)
            upsampled = interpolate_images(lr_images, hr_images.shape[-2:], method)
            upsampled = upsampled.clamp(0.0, 1.0)

            for prediction, target in zip(upsampled, hr_images):
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

    return summarize_metric_values(values)


def format_mean_std(metric_values, precision=3):
    """Format metric mean and standard deviation"""
    return (
        f"{metric_values['mean']:.{precision}f} +/- "
        f"{metric_values['std']:.{precision}f}"
    )


def print_results_table(results):
    """
    Print interpolation metrics as a comparison table.

    Args:
        results: Metrics grouped by method and split.
    """
    print("\n" + "=" * 96)
    print("Performance comparison for interpolation baselines")
    print("=" * 96)
    print(
        f"{'Method':<16}{'Split':<8}{'BACC':<18}{'IIEE (x10^5)':<18}"
        f"{'MAE':<18}{'PSNR':<18}{'SSIM':<18}"
    )
    print("-" * 96)

    for method, method_results in results.items():
        for idx, split_name in enumerate(("Train", "Val", "Test")):
            method_label = method if idx == 0 else ""
            split_results = method_results[split_name.lower()]
            print(
                f"{method_label:<16}{split_name:<8}"
                f"{format_mean_std(split_results['BACC']):<18}"
                f"{format_mean_std(split_results['IIEE']):<18}"
                f"{format_mean_std(split_results['MAE']):<18}"
                f"{format_mean_std(split_results['PSNR'], precision=2):<18}"
                f"{format_mean_std(split_results['SSIM']):<18}"
            )
        print("-" * 96)


def save_metrics(results):
    """
    Save interpolation metrics to outputs/metrics.

    Args:
        results: Metrics grouped by method and split.

    Returns:
        Path to the saved metrics file.
    """
    from config import OUTPUT_METRICS_DIR

    OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_METRICS_DIR / "interpolation_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return metrics_path


def evaluate_interpolation_methods(
    osisaf_dir,
    masam2_dir,
    train_date_range,
    val_date_range,
    test_date_range,
    batch_size=8,
    num_workers=2,
    threshold=0.15,
    methods=INTERPOLATION_METHODS,
    with_missed=False,
):
    """
    Evaluate interpolation baselines on train, validation, and test splits.

    Args:
        osisaf_dir: Directory with OSISAF .npy files.
        masam2_dir: Directory with MASAM2 .npy files.
        train_date_range: Training range as start and end dates.
        val_date_range: Validation range as start and end dates.
        test_date_range: Test range as start and end dates.
        batch_size: Evaluation batch size.
        num_workers: Number of DataLoader workers.
        threshold: Ice/no-ice threshold.
        methods: Interpolation methods to evaluate.
        with_missed: Whether to keep MASAM2 missed-date pairs.

    Returns:
        Metrics grouped by method and split.
    """
    from dataset import create_dataloaders

    print("=" * 70)
    print("EVALUATION OF INTERPOLATION METHODS FOR UPSCALING")
    print("=" * 70)

    train_loader, val_loader, test_loader, _ = create_dataloaders(
        osisaf_dir=osisaf_dir,
        masam2_dir=masam2_dir,
        train_date_range=train_date_range,
        val_date_range=val_date_range,
        test_date_range=test_date_range,
        batch_size=batch_size,
        num_workers=num_workers,
        with_missed=with_missed,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    results = {}
    for method in methods:
        print(f"\nEvaluating method: {method.upper()}")
        results[method] = {
            split_name: evaluate_interpolation_split(
                method=method,
                dataloader=dataloader,
                split_name=split_name,
                threshold=threshold,
            )
            for split_name, dataloader in dataloaders.items()
        }

    print_results_table(results)
    metrics_path = save_metrics(results)
    print(f"Saved metrics: {metrics_path}")
    return results


def main():
    """Parse CLI arguments and run interpolation evaluation"""
    from config import MASAM2_DIR, OSISAF_DIR

    parser = argparse.ArgumentParser(
        description="Evaluate interpolation baselines on train/val/test date splits."
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
        "--methods",
        type=parse_methods,
        default=INTERPOLATION_METHODS,
        help="Comma-separated methods: bilinear,bicubic,nearest.",
    )
    parser.add_argument(
        "--with-missed",
        action="store_true",
        help="Include MASAM2 dates listed in masam2_missed.txt.",
    )

    args = parser.parse_args()

    start_time = time.time()
    results = evaluate_interpolation_methods(
        osisaf_dir=OSISAF_DIR,
        masam2_dir=MASAM2_DIR,
        train_date_range=args.train_range,
        val_date_range=args.val_range,
        test_date_range=args.test_range,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        methods=args.methods,
        with_missed=args.with_missed,
    )

    elapsed_time = time.time() - start_time
    print(f"\nTotal evaluation time: {elapsed_time:.1f} seconds")
    return results


if __name__ == "__main__":
    metrics = main()
