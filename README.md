# Deep Learning for Super-resolution of Sea Ice Concentration: Case Study of OSISAF Downscaler

The trained model (unet.pth) is located in the root directory.

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

### Training the Model
To train the U-Net model from scratch:

```bash
uv run python src/unet_train.py
```

## Visualizing Results

To generate comparison plots between Low Res, High Res, and Prediction:

```bash
uv run python src/visualize_pairs.py
```

## Benchmarking

To evaluate classical interpolation baselines:

```bash
uv run python src/interpolation_methods.py
```
