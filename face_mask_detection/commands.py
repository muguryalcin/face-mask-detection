from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from face_mask_detection.dvc_utils import ensure_data_available, track_models_with_dvc
from face_mask_detection.infer import infer as run_infer
from face_mask_detection.train import train as run_train

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_cfg(overrides=None):
    # Load the Hydra config and apply overrides
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=_parse_overrides(overrides))


def _parse_overrides(overrides):
    # Parse the overrides
    if not overrides:
        return []
    if isinstance(overrides, str):
        return [x.strip() for x in overrides.split(",") if x.strip()]
    if len(overrides) == 1 and isinstance(overrides[0], (list, tuple)):
        overrides = overrides[0]
    return [str(x) for x in overrides]


def check_config(*overrides):
    cfg = _load_cfg(overrides)
    print(OmegaConf.to_yaml(cfg, resolve=True))


def ensure_data(*overrides):
    cfg = _load_cfg(overrides)
    dataset_path = ensure_data_available(cfg)
    print(dataset_path)


def train(*overrides):
    cfg = _load_cfg(overrides)
    run_train(cfg)


def track_models(*overrides):
    cfg = _load_cfg(overrides)
    track_models_with_dvc(cfg)


def infer(image_path, checkpoint_path=None, overrides=None):
    cfg = _load_cfg(overrides)
    run_infer(cfg, image_path=image_path, checkpoint_path=checkpoint_path)


def main():
    fire.Fire(
        {
            "check-config": check_config,
            "ensure-data": ensure_data,
            "ensure_data": ensure_data,
            "data-path": ensure_data,
            "track-models": track_models,
            "train": train,
            "infer": infer,
        }
    )


if __name__ == "__main__":
    main()
