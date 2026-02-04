import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OSISAF_DIR = "D:/dataset_ML4RS_2026/dataset_masam_osisaf/OSISAF"
MASAM2_DIR = "D:/dataset_ML4RS_2026/dataset_masam_osisaf/MASAM2"