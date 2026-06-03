# Face Mask Detection

## Project Description

This repository is an Python package for an end-to-end face mask object detection pipeline. It uses `uv` for package management, DVC for data/model artifact versioning, Hydra/OmegaConf for configuration, MLflow for experiment tracking, PyTorch Lightning for training, and pre-commit for code quality.

## Problem

The goal of this project is to build a computer vision system that detects faces and predicts whether each person is wearing a mask correctly, incorrectly, or not at all. This can be useful for healthcare, public safety, and manufacturing settings where mask checks may be needed.

## Inputs And Outputs

Training inputs are RGB images of varying sizes with YOLO mask bounding boxes and three foreground labels:

```text
with_mask
without_mask
mask_weared_incorrect
```

Inference input is a public RGB image containing faces. Model outputs are bounding box coordinates, a predicted class, and a confidence score for each detected face.

## Data

The project uses the Kaggle Face Mask Detection dataset: `asemsaber/face-mask-detection`.

Key dataset details:

- Size: about 2.18 GB.
- Images: 1706 RGB images.
- Classes: `with_mask`, `without_mask`, `mask_weared_incorrect`.
- Annotations: YOLO and Pascal VOC formats.
- License: CC0 Public Domain.
- Class imbalance by annotated faces: `with_mask` 6464 (79.37%), `without_mask` 1434 (17.61%), `mask_weared_incorrect` 246 (3.02%).

The data is COVID-era and may not fully generalize to all real deployment settings. Images include varied sizes, occlusions, and crowding.

## Validation Split

The original dataset split is roughly 63/27/10, but it can leak because original and augmented versions of the same image are in different splits. This project uses a stratified group split by original image ID, targeting 70/15/15 for train/validation/test with a fixed seed. Grouping keeps an original image and its augmented variants in the same split.

## Modeling

The baseline and main model is Faster R-CNN with a ResNet50 FPN backbone. Later improvements could include more careful augmentation, class imbalance handling, fine-tuning, larger datasets, or a better detector architecture.

Expected object detection metrics are rough targets based on comparable Kaggle notebooks, not guaranteed production results:

- `mAP@0.5`: around 0.65.
- `mAP@0.95`: around 0.4.
- Per-class AP: around 0.75+.
- Precision: around 0.9.
- Recall: around 0.55.

## Setup

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or check the [uv installation docs](https://astral.sh/uv/usage/install/) for other platforms and package manager options.

Clone and install the package with development dependencies:

```bash
git clone https://github.com/muguryalcin/face-mask-detection.git
cd face-mask-detection
uv sync --dev
```

You can check the current configuratio and command setup with:

```bash
uv run detect-mask check-config
# Or with overrides:
uv run python -m face_mask_detection.commands check-config training.max_epochs=10
```

## Data Retrieval

The repository already configures two local DVC remotes in `.dvc/config`:

```text
data_remote: ../dvc-storage/data
models_remote: ../dvc-storage/models
```

For a clean clone on the same machine, create the configured local remote folders and run the `ensure-data` command to download the data:

```bash
mkdir -p ../dvc-storage/data ../dvc-storage/models
uv run detect-mask ensure-data
```

Training, split regeneration, and inference all call the `ensure-data` check. It downloads the Kaggle dataset, splits it and fixes the dataset layout.

## Split Regeneration

`split-data` splits the dataset into train/validation/test with a stratified group split by original image ID to prevent leakage. If you want to regenerate the splits with a different random seed or split ratio, change the ratios in the config then run:

```bash
uv run detect-mask split-data
```

## Pre-Commit

Install hooks:

```bash
uv run pre-commit install
```

Run all checks:

```bash
uv run pre-commit run -a
```

## Configuration

Important config files:

```text
configs/config.yaml
configs/data/face_mask.yaml
configs/model/fasterrcnn_resnet50_fpn.yaml
configs/preprocessing/default.yaml
configs/training/default.yaml
configs/training/h100.yaml
configs/training/smoke.yaml
configs/inference/default.yaml
configs/logging/mlflow.yaml
```

Hydra overrides can be passed after the command. Example:

```bash
uv run detect-mask train training=smoke model.pretrained=false
```

The default model config uses `model.pretrained=true`, which may download torchvision weights. For a safe smoke test without external weight download, use `model.pretrained=false`.

## MLflow

Start an MLflow server in a separate terminal if you want experiment tracking through the configured local server:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

The MLflow configuration is in `configs/logging/mlflow.yaml`.

## Training

Run default training:

```bash
uv run detect-mask train
```

Or for a quick smoke test without downloading pretrained weights:

```bash
uv run detect-mask train training=smoke model.pretrained=false
```

Training first ensures data is available through DVC or Kaggle fallback, then writes checkpoints, model state dicts, plots, and MLflow artifacts.

## Inference

Run inference on one image:

```bash
uv run detect-mask infer --image-path <path-to-image> --checkpoint-path <path-to-checkpoint>
```

Example inputs after data/model retrieval or training:

```text
data/dataset/test/images/<image-name>.png
checkpoints/last.ckpt
```

Inference ensures data is available, loads the configured Faster R-CNN model or provided checkpoint, applies configured preprocessing including normalization, and prints detections above `configs/inference/default.yaml` score threshold.

## Artifacts

Training can generate:

```text
checkpoints/              Lightning checkpoints, including last.ckpt
models/                   Saved model state dicts, including last_state_dict.pt
plots/train_loss.png      Training loss curve
plots/val_loss.png        Validation loss curve
plots/loss_components.png Loss component curves
mlruns/                   Local MLflow runs if using a local file store
```

Track and push model artifacts to the configured local DVC model remote:

```bash
uv run detect-mask track-models
uv run dvc push -r models_remote models.dvc
```

## Repository Structure

```text
configs/                  Hydra configuration files
face_mask_detection/      Installable Python package
  commands.py             Fire CLI entry point for detect-mask
  data.py                 Dataset and LightningDataModule
  dvc_utils.py            DVC pull and Kaggle download fallback helpers
  infer.py                Single-image inference
  metrics.py              Training metric history callback
  model.py                Faster R-CNN LightningModule
  plots.py                Training plot generation
  split_data.py           Leakage-safe split regeneration
data.dvc                  DVC-tracked data artifact
pyproject.toml            Package metadata and dependencies
uv.lock                   Locked dependency graph
```
