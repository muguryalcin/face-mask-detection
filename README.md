# Face Mask Detection

## Project Description

This project is an end-to-end machine learning pipeline for face mask detection. It includes data preparation, model training, evaluation and inference. The goal of this project is to build a computer vision system that detects faces and if people are wearing masks or not. This is useful for healthcare, public safety and some manufacturing factories. This project aims to be an end-to-end deployable ML system, which would make it easy to use when needed. It uses DVC for data versioning, MLFlow for experiment tracking, pre-commit hooks for the code quality and uv for package management and task running, Hydra and OmegaConf for configuration management.

The dataset used in this project is a collection of images of people wearing masks, not wearing masks and wearing masks incorrectly. The model is trained to classify these three classes. FasterRCNN_ResNet50_FPN is used as the base model architecture. The project is structured to allow easy experimentation and extension with different models, datasets and configurations.

## Setup

This project uses `uv` for package management and task running. If you don't have `uv` installed, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

check the [uv documentation](https://astral.sh/uv/) for more details.

Starting with the project setup:

```bash
git clone https://github.com/muguryalcin/face-mask-detection.git
cd face-mask-detection
uv sync --dev
```

Pull data with DVC from the remote storage (Google Drive in this case):

```bash
uv run dvc pull
```

# Check Dataset

Prepare/check dataset from the CLI:

```bash
uv run detect-mask ensure_data
```

## Pre-commit Hooks

Install pre-commit hooks that are used in the project:

```bash
uv run pre-commit install
```

You can run pre-commit checks on all files with:

```bash
uv run pre-commit run -a
```

# Configurations

Dataset config:

```text
configs/data/face_mask.yaml
```

Main config:

```text
configs/config.yaml
```

## MLflow

To track the experiments, you need to start the MLFlow server:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Open in your browser:

```text
http://127.0.0.1:8080
```

You can change the MLFlow logging configuration in:

```text
configs/logging/mlflow.yaml
```

## Training

Default training command (it uses the default configurations specified in `configs/config.yaml`):

```bash
uv run detect-mask train
```

Smoke training run (for testing the training pipeline, it runs for 1 epoch and uses only 1 batch for training and validation):

```bash
uv run detect-mask train model.pretrained=false training.max_epochs=1 training.limit_train_batches=1 training.limit_val_batches=1 training.batch_size=1 training.num_workers=0
```

## Artifacts

You can see the generated artifacts after running the train command.

```text
checkpoints/
models/
plots/
mlruns/
```

Plots generated after training:

```text
plots/train_loss.png # Training loss curve
plots/val_loss.png # Validation loss curve
plots/loss_components.png # Loss components curve (classification loss, box regression loss, etc.)
```

## Inference

You can run inference on a single image with the following command:

```bash
uv run detect-mask infer --image-path <path-to-image> --checkpoint-path <path-to-checkpoint>
```

Example image path:

```text
data/dataset/test/images/<image-name>.png
```

Example checkpoint path (after training, the best and last checkpoint will be saved in the `checkpoints/` directory):

```text
checkpoints/last.ckpt
```

## DVC

Pull artifacts:

```bash
uv run dvc pull
```

Push data changes if needed:

```bash
uv run dvc add data
uv run dvc push
```

## Repository Structure

```text
configs/
face_mask_detection/
  cli.py
  data.py
  dvc_utils.py
  metrics.py
  model.py
  plots.py
  train.py
plots/
pyproject.toml
uv.lock
data.dvc
```
