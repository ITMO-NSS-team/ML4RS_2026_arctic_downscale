import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import matplotlib
import torch
import torch.nn.functional as F

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

matplotlib.use("Agg")

from config import (
    ATTENTION_UNET_PREDICTIONS_DIR,
    DEVICE,
    LIGHT_UNET_PREDICTIONS_DIR,
    MASAM2_DIR,
    OSISAF_DIR,
    OUTPUT_MODELS_DIR,
)
from dataset import IceConcentrationDataset
from visualization.visualization import visualize_results


MODEL_MODULES = {
    "unet_light": "models.unet_light",
    "attention_unet": "models.attention_unet",
}

MODEL_CLASSES = {
    "unet_light": "UNetLight",
    "attention_unet": "AttentionUNet",
}

DEFAULT_WEIGHTS = {
    "unet_light": OUTPUT_MODELS_DIR / "unet_light_final.pth",
    "attention_unet": OUTPUT_MODELS_DIR / "attention_unet_final.pth",
}

DEFAULT_OUTPUT_DIRS = {
    "unet_light": LIGHT_UNET_PREDICTIONS_DIR,
    "attention_unet": ATTENTION_UNET_PREDICTIONS_DIR,
}


def validate_date(value):
    """Validate a CLI date string."""
    if len(value) != 8 or not value.isdigit():
        raise argparse.ArgumentTypeError("Date must use YYYYMMDD format")
    return value


def load_model(model_name, weights_path):
    """
    Load a prediction model and weights.

    Args:
        model_name: Model key from MODEL_MODULES.
        weights_path: Path to weights or checkpoint.

    Returns:
        Model in evaluation mode.
    """
    module = importlib.import_module(MODEL_MODULES[model_name])
    model_class = getattr(module, MODEL_CLASSES[model_name])
    model = model_class(in_channels=1, out_channels=1).to(DEVICE)

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def find_sample_by_date(dataset, date):
    """
    Find a dataset sample for a specific date.

    Args:
        dataset: IceConcentrationDataset instance.
        date: Date in YYYYMMDD format.

    Returns:
        Sample dictionary for the requested date.
    """
    for idx, (osisaf_path, _) in enumerate(dataset.pairs):
        sample_date = dataset._extract_date(Path(osisaf_path).name)
        if sample_date == date:
            return dataset[idx]

    raise ValueError(f"No OSISAF/MASAM2 pair found for date {date}")


def predict_by_date(
    model_name,
    date,
    weights_path=None,
    output_dir=None,
    osisaf_dir=OSISAF_DIR,
    masam2_dir=MASAM2_DIR,
):
    """
    Run one-date prediction and save array and figure outputs.

    Args:
        model_name: Model key from MODEL_MODULES.
        date: Prediction date in YYYYMMDD format.
        weights_path: Optional path to weights or checkpoint.
        output_dir: Optional directory for prediction outputs.
        osisaf_dir: Directory with OSISAF .npy files.
        masam2_dir: Directory with MASAM2 .npy files.

    Returns:
        Paths to the saved prediction array and figure.
    """
    weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS[model_name]
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIRS[model_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    dataset = IceConcentrationDataset(osisaf_dir=osisaf_dir, masam2_dir=masam2_dir)
    sample = find_sample_by_date(dataset, date)

    lr_image = sample["lr"].unsqueeze(0).to(DEVICE)
    hr_image = sample["hr"].unsqueeze(0).to(DEVICE)

    model = load_model(model_name, weights_path)

    with torch.no_grad():
        prediction = model(lr_image).clamp(0.0, 1.0)
        if prediction.shape[-2:] != hr_image.shape[-2:]:
            prediction = F.interpolate(
                prediction,
                size=hr_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

    prediction_np = prediction.squeeze().cpu().numpy().astype(np.float32)
    prediction_path = output_dir / f"{date}.npy"
    figure_path = output_dir / f"{date}.png"
    np.save(prediction_path, prediction_np)

    visualize_results(
        lr_images=lr_image.cpu(),
        hr_images=hr_image.cpu(),
        pred_images=prediction.cpu(),
        epoch="prediction",
        model_name=f"{model_name} Super Resolution",
        save_path=str(figure_path),
        dates=[date],
    )

    mae = torch.mean(torch.abs(prediction - hr_image)).item()

    print(f"Model: {model_name}")
    print(f"Date: {date}")
    print(f"Weights: {weights_path}")
    print(f"OSISAF: {sample['lr_path']}")
    print(f"MASAM2: {sample['hr_path']}")
    print(f"MAE vs MASAM2: {mae:.6f}")
    print(f"Saved prediction: {prediction_path}")
    print(f"Saved visualization: {figure_path}")

    return prediction_path, figure_path


def main():
    """Parse CLI arguments and run one-date prediction"""
    parser = argparse.ArgumentParser(
        description="Run a trained U-Net model for a single date."
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_MODULES.keys()),
        required=True,
        help="Model architecture.",
    )
    parser.add_argument(
        "--date",
        type=validate_date,
        required=True,
        help="Prediction date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to .pth weights. Defaults to outputs/models/<model>_final.pth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved .npy prediction and .png visualization.",
    )
    parser.add_argument(
        "--osisaf-dir",
        default=OSISAF_DIR,
        help="Path to OSISAF .npy directory.",
    )
    parser.add_argument(
        "--masam2-dir",
        default=MASAM2_DIR,
        help="Path to MASAM2 .npy directory.",
    )

    args = parser.parse_args()

    predict_by_date(
        model_name=args.model,
        date=args.date,
        weights_path=args.weights,
        output_dir=args.output_dir,
        osisaf_dir=args.osisaf_dir,
        masam2_dir=args.masam2_dir,
    )


if __name__ == "__main__":
    main()
