from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OSISAF_DIR = "./data/OSISAF"
MASAM2_DIR = "./data/MASAM2"

LIGHT_MODEL_PATH = PROJECT_ROOT / "models" / "unet_light_final.pth"
