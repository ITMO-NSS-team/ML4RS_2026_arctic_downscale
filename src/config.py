import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OSISAF_DIR = "./data/OSISAF"
MASAM2_DIR = "./data/MASAM2"

MODEL_PATH = "../models/unet_final_model.pth"
