# Deep Learning for Super-resolution of Sea Ice Concentration: Case Study of OSISAF Downscaler

The lightweight U-Net checkpoint used in the paper is located at
`outputs/models/unet_light_checkpoint.pth`.

## Installation

1.  Clone the repository:

```bash
git clone <repository_url>
cd <repository_name>
```

2.  Create a virtual environment and sync dependencies:

```bash
uv sync
source .venv/bin/activate
```

## Usage

Ensure your data is placed in `./data/OSISAF` and `./data/MASAM2` or update `src/config.py`.
When restored MASAM2 dates must be excluded, place `masam2_missed.txt` next
to those directories:

```text
data/
├── OSISAF/
├── MASAM2/
└── masam2_missed.txt
```

### Training the Lightweight Model

To train the lightweight U-Net model from the paper:

```bash
uv run python src/training/unet_light_train.py
```

Training data is split by three sequential inclusive date ranges in
`YYYYMMDD` format. Update these ranges in `src/training/unet_light_train.py` for your
dataset:

```python
train_loader, val_loader, test_loader, full_dataset = create_dataloaders(
    osisaf_dir=OSISAF_DIR,
    masam2_dir=MASAM2_DIR,
    batch_size=8,
    train_date_range=("20120701", "20201231"),
    val_date_range=("20210101", "20221231"),
    test_date_range=("20230101", "20250630"),
    with_missed=False,
)
```

With `with_missed=False`, pairs whose dates are listed in
`masam2_missed.txt` are excluded from all three splits. Set it to `True` to
include the restored MASAM2 matrices.

## Visualizing Results

Visualization helpers are located in `src/visualization/visualization.py`.
Generated figures are saved under `outputs/figures/`.

## Predicting One Date

To run a trained model for a selected date:

```bash
uv run python src/inference/predict_by_date.py --model unet_light --date 20230101
```

Available models are `unet_light` and `attention_unet`. By default, weights are
loaded from `outputs/models/unet_light_final.pth` or `outputs/models/attention_unet_final.pth`.
Use `--weights` to pass a custom checkpoint.

## Benchmarking

To evaluate classical interpolation baselines:

```bash
uv run python src/inference/interpolation_methods.py
```
