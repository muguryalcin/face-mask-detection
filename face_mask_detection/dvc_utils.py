import shutil
import subprocess
import sys
from importlib import import_module
from pathlib import Path

from omegaconf import OmegaConf


def ensure_data_available(cfg):
    # Ensures the dataset is available locally, either by pulling with DVC or downloading from Kaggle.
    data_root = _data_root(cfg)

    # Pull from the DVC
    if not (data_root / cfg.data.data_yaml).exists():
        subprocess.run(
            [sys.executable, "-m", "dvc", "pull"], cwd=_repo_root(), check=False
        )

    # If still not available, download from Kaggle and normalize the layout.
    if not (data_root / cfg.data.data_yaml).exists():
        _download_from_kaggle(cfg, data_root)

    # Fix the folder structure and validate
    _normalize_kaggle_layout(data_root, cfg.data.data_yaml)
    dataset_dir = _dataset_dir(data_root, cfg.data.data_yaml)
    _validate_splits(dataset_dir)
    return dataset_dir


def _data_root(cfg):
    # Determine the absolute path to the root.
    data_root = Path(cfg.paths.data_root)
    if data_root.is_absolute():
        return data_root
    return _repo_root() / data_root


def _dataset_dir(data_root, data_yaml_name):
    # Get the dataset directory from the data.yaml file.
    data_yaml = OmegaConf.load(data_root / data_yaml_name)
    return (data_root / str(data_yaml.path).replace("\\", "/")).resolve()


def _download_from_kaggle(cfg, data_root):
    # Download the dataset from Kaggle using the kagglehub package.
    kagglehub = import_module("kagglehub")
    downloaded_path = Path(kagglehub.dataset_download(cfg.data.kaggle_dataset))

    # Clear the existing data root if it exists, then copy the downloaded dataset there.
    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(downloaded_path, data_root)


def _normalize_kaggle_layout(data_root, data_yaml_name):
    # Flatten the dataset's nested folder structure.
    dataset_dir = _dataset_dir(data_root, data_yaml_name)
    flat_dataset_dir = data_root / "dataset"

    # If the dataset is already in the correct location, do nothing. Otherwise, move it to the expected path.
    if dataset_dir != flat_dataset_dir.resolve():
        if flat_dataset_dir.exists():
            shutil.rmtree(flat_dataset_dir)
        shutil.move(str(dataset_dir), flat_dataset_dir)

    # Remove any redundant folders that may have been created during the download process.
    repeated_folder = data_root / "Face Mask Detection"
    if repeated_folder.exists():
        shutil.rmtree(repeated_folder)

    # Update the data.yaml to point to the new dataset location.
    data_yaml = OmegaConf.load(data_root / data_yaml_name)
    data_yaml.path = "dataset"
    OmegaConf.save(data_yaml, data_root / data_yaml_name)


def _validate_splits(dataset_dir):
    # Ensure that the expected dataset splits (train, valid, test) are present in the dataset directory.
    missing = [
        split
        for split in ("train", "valid", "test")
        if not (dataset_dir / split).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"Missing dataset splits {missing} in {dataset_dir}")


def _repo_root():
    return Path(__file__).resolve().parents[1]
