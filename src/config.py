from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OSISAF_DIR = "./data/OSISAF"
MASAM2_DIR = "./data/MASAM2"
OSISAF_REPROJECTED_DIR = "./data/OSISAF_reproj_matrices"
MASIE_REPROJECTED_DIR = "./data/MASIE_reproj_matrices"
MANUAL_HYBRID_REPROJECTED_DIR = "./data/ManualHybrid_reproj_matrices"
MASAM2_LANDMASK_PATH = "./data/MASAM2_landmask.npy"

OUTPUT_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUT_PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
OUTPUT_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUT_METRICS_DIR = OUTPUTS_DIR / "metrics"
LIGHT_UNET_PREDICTIONS_DIR = OUTPUT_PREDICTIONS_DIR / "light_unet"
ATTENTION_UNET_PREDICTIONS_DIR = OUTPUT_PREDICTIONS_DIR / "attention_unet"
MAP_COMPARISON_FIGURES_DIR = OUTPUT_FIGURES_DIR / "prediction_comparison_images"

LIGHT_MODEL_PATH = OUTPUT_MODELS_DIR / "unet_light_final.pth"
