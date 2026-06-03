from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from face_mask_detection.dvc_utils import ensure_data_available
from face_mask_detection.infer import infer as run_infer
from face_mask_detection.train import train as run_train

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_cfg(overrides=None):
    # Loads the config and applies overrides
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="config", overrides=_parse_overrides(overrides))


def _parse_overrides(overrides):
    # Applies the overrides
    if not overrides:
        return []
    if isinstance(overrides, str):
        return [x.strip() for x in overrides.split(",") if x.strip()]
    return [str(x) for x in overrides]


def check_config(overrides=None):
    # Loads the config and prints it -- debug purposes
    cfg = _load_cfg(overrides)
    print(OmegaConf.to_yaml(cfg, resolve=True))


def data_path(overrides=None):
    # Ensures the data is available and prints the path
    cfg = _load_cfg(overrides)
    dataset_path = ensure_data_available(cfg)
    print(dataset_path)


def train(overrides=None):
    # Train the model using the config.
    # load the config
    cfg = _load_cfg(overrides)
    run_train(cfg)


def infer(image_path, checkpoint_path=None, overrides=None):
    # Run inference
    cfg = _load_cfg(overrides)
    run_infer(cfg, image_path=image_path, checkpoint_path=checkpoint_path)


def main():
    fire.Fire(
        {
            "check-config": check_config,
            "data-path": data_path,
            "train": train,
            "infer": infer,
        }
    )


if __name__ == "__main__":
    main()
