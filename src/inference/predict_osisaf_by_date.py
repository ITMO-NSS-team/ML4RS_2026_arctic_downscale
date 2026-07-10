import argparse
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import DEVICE, MODEL_CONFIGS, OSISAF_DIR


PREDICTION_MODEL_CONFIGS = {
    name: config for name, config in MODEL_CONFIGS.items()
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
        model_name: Model key from MODEL_CONFIGS.
        weights_path: Path to weights or checkpoint.

    Returns:
        Model in evaluation mode.
    """
    model_config = PREDICTION_MODEL_CONFIGS[model_name]
    module = importlib.import_module(model_config["module"])
    model_class = getattr(module, model_config["class_name"])
    model = model_class(in_channels=1, out_channels=1).to(DEVICE)

    checkpoint = torch.load(weights_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def extract_date(filename, date_pattern=r"(\d{8})"):
    """
    Extract an YYYYMMDD date from a filename.

    Args:
        filename: Filename to inspect.
        date_pattern: Regex for extracting dates from filenames.

    Returns:
        Date string, or None if no date is found.
    """
    match = re.search(date_pattern, filename)
    return match.group(1) if match else None


def find_osisaf_path(osisaf_dir, date):
    """
    Find an OSISAF matrix for a specific date.

    Args:
        osisaf_dir: Directory with OSISAF .npy files.
        date: Date in YYYYMMDD format.

    Returns:
        Path to the OSISAF matrix.
    """
    matches = sorted(
        path
        for path in Path(osisaf_dir).rglob("*.npy")
        if extract_date(path.name) == date
    )

    if not matches:
        raise FileNotFoundError(f"No OSISAF .npy file found for date {date}")

    return matches[0]


def preprocess_osisaf(data):
    """
    Normalize OSISAF concentrations stored as percentages to [0, 1].

    Args:
        data: Raw OSISAF concentration matrix.

    Returns:
        Normalized matrix.
    """
    data = np.clip(data, 0, 100).astype(np.float32)
    return data / 100.0


def load_osisaf_tensor(osisaf_path):
    """
    Load one OSISAF matrix and convert it to a model input tensor.

    Args:
        osisaf_path: Path to an OSISAF .npy file.

    Returns:
        Input tensor with shape (1, 1, H, W).
    """
    osisaf = np.load(osisaf_path)
    if osisaf.ndim != 2:
        raise ValueError(f"OSISAF matrix must be two-dimensional: {osisaf_path}")

    osisaf = preprocess_osisaf(osisaf)
    return torch.from_numpy(osisaf).unsqueeze(0).unsqueeze(0).float()


def predict_osisaf_by_date(
    model_name,
    date,
    weights_path=None,
    output_dir=None,
    osisaf_dir=OSISAF_DIR,
):
    """
    Run one-date prediction from OSISAF only and save output files.

    Args:
        model_name: Model key from MODEL_CONFIGS.
        date: Prediction date in YYYYMMDD format.
        weights_path: Optional path to weights or checkpoint.
        output_dir: Optional directory for prediction outputs.
        osisaf_dir: Directory with OSISAF .npy files.

    Returns:
        Path to the saved prediction array.
    """
    model_config = PREDICTION_MODEL_CONFIGS[model_name]
    weights_path = Path(weights_path) if weights_path else model_config["final_path"]
    output_dir = Path(output_dir) if output_dir else model_config["prediction_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    osisaf_path = find_osisaf_path(osisaf_dir, date)
    lr_image = load_osisaf_tensor(osisaf_path).to(DEVICE)
    model = load_model(model_name, weights_path)

    with torch.no_grad():
        prediction = model(lr_image).clamp(0.0, 1.0)

    prediction_np = prediction.squeeze().cpu().numpy().astype(np.float32)
    prediction_path = output_dir / f"{date}.npy"
    np.save(prediction_path, prediction_np)

    print(f"Model: {model_name}")
    print(f"Date: {date}")
    print(f"Weights: {weights_path}")
    print(f"OSISAF: {osisaf_path}")
    print(f"Saved prediction: {prediction_path}")

    return prediction_path


def main():
    """Parse CLI arguments and run one-date OSISAF-only prediction."""
    parser = argparse.ArgumentParser(
        description="Run a trained U-Net model for a single OSISAF date."
    )
    parser.add_argument(
        "--model",
        choices=tuple(PREDICTION_MODEL_CONFIGS.keys()),
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
        help="Directory for saved .npy prediction.",
    )
    parser.add_argument(
        "--osisaf-dir",
        type=Path,
        default=OSISAF_DIR,
        help="Path to OSISAF .npy directory.",
    )
    args = parser.parse_args()

    predict_osisaf_by_date(
        model_name=args.model,
        date=args.date,
        weights_path=args.weights,
        output_dir=args.output_dir,
        osisaf_dir=args.osisaf_dir,
    )


if __name__ == "__main__":
    main()
