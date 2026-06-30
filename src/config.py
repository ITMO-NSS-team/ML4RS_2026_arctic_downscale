from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OSISAF_DIR = DATA_DIR / "OSISAF"
MASAM2_DIR = DATA_DIR / "MASAM2"
MASAM2_RAW_DIR = DATA_DIR / "MASAM2_raw"
MASAM2_MASK_NETCDF_PATH = MASAM2_DIR / "masam2.201207.nc"
MASIE_TARGET_GRID_PATH = DATA_DIR / "MASIE_fix" / "20230101.nc"
OSISAF_REPROJECTED_DIR = DATA_DIR / "OSISAF_reproj_matrices"
MASIE_REPROJECTED_DIR = DATA_DIR / "MASIE_reproj_matrices"
MANUAL_HYBRID_REPROJECTED_DIR = DATA_DIR / "ManualHybrid_reproj_matrices"
MASAM2_LANDMASK_PATH = DATA_DIR / "MASAM2_landmask.npy"
MASAM2_MISSED_DATES_PATH = DATA_DIR / "masam2_missed.txt"
OSISAF_REPROJECT_CROP_BOUNDS = (1907, 1907 + 2550, 2000, 2000 + 2100)

OUTPUT_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUT_PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
OUTPUT_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUT_METRICS_DIR = OUTPUTS_DIR / "metrics"
LIGHT_UNET_PREDICTIONS_DIR = OUTPUT_PREDICTIONS_DIR / "light_unet"
ATTENTION_UNET_PREDICTIONS_DIR = OUTPUT_PREDICTIONS_DIR / "attention_unet"
MAP_COMPARISON_FIGURES_DIR = OUTPUT_FIGURES_DIR / "prediction_comparison_images"

LIGHT_UNET_BEST_MODEL_PATH = OUTPUT_MODELS_DIR / "unet_light_best.pth"
LIGHT_UNET_FINAL_MODEL_PATH = OUTPUT_MODELS_DIR / "unet_light_final.pth"
ATTENTION_UNET_BEST_MODEL_PATH = OUTPUT_MODELS_DIR / "attention_unet_best.pth"
ATTENTION_UNET_FINAL_MODEL_PATH = OUTPUT_MODELS_DIR / "attention_unet_final.pth"

TRAIN_DATE_RANGE = ("20120701", "20201231")
VAL_DATE_RANGE = ("20210101", "20221231")
TEST_DATE_RANGE = ("20230101", "20250630")

MODEL_CONFIGS = {
    "unet_light": {
        "module": "models.unet_light",
        "class_name": "UNetLight",
        "best_path": LIGHT_UNET_BEST_MODEL_PATH,
        "final_path": LIGHT_UNET_FINAL_MODEL_PATH,
        "prediction_dir": LIGHT_UNET_PREDICTIONS_DIR,
    },
    "attention_unet": {
        "module": "models.attention_unet",
        "class_name": "AttentionUNet",
        "best_path": ATTENTION_UNET_BEST_MODEL_PATH,
        "final_path": ATTENTION_UNET_FINAL_MODEL_PATH,
        "prediction_dir": ATTENTION_UNET_PREDICTIONS_DIR,
    },
}
