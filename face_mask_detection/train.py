import subprocess
from pathlib import Path

import mlflow
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from face_mask_detection.data import FaceMaskDataModule
from face_mask_detection.dvc_utils import ensure_data_available
from face_mask_detection.metrics import MetricsHistory
from face_mask_detection.model import FaceMaskDetector
from face_mask_detection.plots import write_training_plots


def train(cfg):
    # Seed everything for reproducibility and create the output dirs.
    seed_everything(int(cfg.seed), workers=True)
    for path in (cfg.paths.plots_dir, cfg.paths.models_dir, cfg.paths.checkpoints_dir):
        Path(path).mkdir(parents=True, exist_ok=True)

    # Make sure the data is available and create data, model modules
    ensure_data_available(cfg)
    datamodule = FaceMaskDataModule(cfg)
    model = FaceMaskDetector(cfg)

    # Create the logger
    logger = _mlflow_logger(cfg)
    _log_extra_params(logger, cfg)

    # Model checkpoint callback to save the best model and the last model based on validation loss.
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(cfg.paths.checkpoints_dir)),
        filename="face-mask-{epoch:02d}",
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    history_callback = MetricsHistory()

    # Create the trainer and start training
    trainer = Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator=str(cfg.training.accelerator),
        devices=int(cfg.training.devices),
        # Use the full dataset if the config value is 1.0 or not set, otherwise use the specified fraction of batches.
        limit_train_batches=cfg.training.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.training.get("limit_val_batches", 1.0),
        log_every_n_steps=int(cfg.training.log_every_n_steps),
        gradient_clip_val=float(cfg.training.gradient_clip_val),
        logger=logger,
        callbacks=[checkpoint_callback, history_callback],
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)

    # Last checkpoint path
    last_checkpoint = Path(checkpoint_callback.last_model_path)
    model_path = Path(cfg.paths.models_dir) / "last_state_dict.pt"
    torch.save(model.state_dict(), model_path)

    # Write training plots and log all artifacts to MLflow
    plot_paths = write_training_plots(
        history_callback.history, Path(cfg.paths.plots_dir)
    )
    _log_artifacts(logger, [last_checkpoint, model_path, *plot_paths])

    # Print summary of saved artifacts
    print(f"Saved checkpoint: {last_checkpoint}")
    print(f"Saved model state dict: {model_path}")
    print("Saved plots:")
    for path in plot_paths:
        print(f"- {path}")


def _mlflow_logger(cfg):
    # Configure MLFlow logger with the tracking URI and experiment name from the config.
    tracking_uri = str(cfg.logging.tracking_uri).strip()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(cfg.logging.experiment_name))

    return MLFlowLogger(
        experiment_name=str(cfg.logging.experiment_name),
        run_name=str(cfg.logging.run_name),
        tracking_uri=tracking_uri,
    )


def _log_extra_params(logger, cfg):
    params = {
        "git_commit": _git_commit_id(),
        "model_name": str(cfg.model.name),
        "image_size": int(cfg.preprocessing.image_size),
        "batch_size": int(cfg.training.batch_size),
        "learning_rate": float(cfg.model.learning_rate),
        "weight_decay": float(cfg.model.weight_decay),
    }
    logger.log_hyperparams(params)


def _log_artifacts(logger, paths):
    experiment = logger.experiment
    run_id = logger.run_id
    for path in paths:
        path = Path(path)
        if path.exists():
            experiment.log_artifact(run_id, str(path))


def _git_commit_id():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()
