from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OSISAF_DIR = "./data/OSISAF"
MASAM2_DIR = "./data/MASAM2"

OUTPUT_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUT_PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
OUTPUT_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUT_METRICS_DIR = OUTPUTS_DIR / "metrics"

LIGHT_MODEL_PATH = OUTPUT_MODELS_DIR / "unet_light_final.pth"
