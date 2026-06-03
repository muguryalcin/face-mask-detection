import logging
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)
KAGGLE_DATASET_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download"


def ensure_data_available(cfg):
    # Ensures the dataset is available locally, either by pulling with DVC or downloading from Kaggle.
    data_root = _data_root(cfg)
    data_yaml_name = str(cfg.data.data_yaml)
    data_yaml_path = data_root / data_yaml_name

    _status(f"Checking dataset at: {data_root}")
    if data_yaml_path.exists():
        _status(f"Dataset YAML found: {data_yaml_path}")
    else:
        _status("Dataset missing, running DVC pull...")
        if _run_dvc(["pull"]):
            _status("DVC pull succeeded")
        else:
            _status("DVC pull failed")

    # If the YAML file still doesn't exist, download from Kaggle.
    if not data_yaml_path.exists():
        _status("DVC pull did not restore data, downloading from Kaggle...")
        _download_from_kaggle(cfg, data_root)

    # Normalize the layout if its not already (necessary for the kaggle download)
    if not _has_canonical_layout(data_root, data_yaml_name):
        _status(f"Normalizing Kaggle layout to {data_root / 'dataset'}")
        _normalize_kaggle_layout(data_root, data_yaml_name)

    # Validate the expected splits are present and return the dataset directory.
    dataset_dir = _dataset_dir(data_root, data_yaml_name)
    _validate_splits(dataset_dir)
    _status(f"Dataset ready at: {dataset_dir}")
    return dataset_dir


def track_models_with_dvc(cfg) -> None:
    models_dir = Path(cfg.paths.models_dir)
    if not models_dir.is_absolute():
        models_dir = _repo_root() / models_dir

    if not models_dir.is_dir() or not any(models_dir.iterdir()):
        _status(f"No model artifacts found at: {models_dir}")
        return

    target = _relative_to_repo(models_dir)
    if _run_dvc(["add", target]):
        dvc_file = f"{target}.dvc"
        _status(f"Model artifacts tracked with DVC: {dvc_file}")
        _status(f"To push models, run: uv run dvc push -r models_remote {dvc_file}")
    else:
        _status("DVC model tracking failed")


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
    dataset = str(cfg.data.kaggle_dataset)
    url = f"{KAGGLE_DATASET_DOWNLOAD_URL}/{dataset}"
    _status(f"Downloading Kaggle dataset with curl: {dataset}")

    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)
        zip_path = temporary_path / "face-mask-detection.zip"
        extract_path = temporary_path / "extracted"

        result = subprocess.run(
            ["curl", "-L", "-o", str(zip_path), url],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            if result.stderr.strip():
                _status(result.stderr.strip())
            raise RuntimeError(
                f"Kaggle download failed with exit code {result.returncode}"
            )

        _status(f"Downloaded Kaggle zip to: {zip_path}")
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zip_file:
            zip_file.extractall(extract_path)
        _status(f"Extracted Kaggle dataset to: {extract_path}")

        if data_root.exists():
            shutil.rmtree(data_root)
        data_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extract_path, data_root)


def _has_canonical_layout(data_root, data_yaml_name):
    # Checks if the dataset is already in the expected layout
    data_yaml_path = data_root / data_yaml_name
    if not data_yaml_path.exists():
        return False

    data_yaml = OmegaConf.load(data_yaml_path)
    if str(data_yaml.path).replace("\\", "/") != "dataset":
        return False
    return (data_root / "dataset").is_dir()


def _normalize_kaggle_layout(data_root, data_yaml_name):
    # Fix the dataset layout
    data_yaml_path = data_root / data_yaml_name
    data_yaml = OmegaConf.load(data_yaml_path)
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

    if str(data_yaml.path).replace("\\", "/") != "dataset":
        data_yaml.path = "dataset"
        OmegaConf.save(data_yaml, data_yaml_path)


def _validate_splits(dataset_dir):
    # Ensure that the expected dataset splits (train, valid, test) are present in the dataset directory.
    missing = [
        split
        for split in ("train", "valid", "test")
        if not (dataset_dir / split).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"Missing dataset splits {missing} in {dataset_dir}")


def _run_dvc(args: list[str]) -> bool:
    command = [sys.executable, "-m", "dvc", *args]
    _status(f"Running: uv run dvc {' '.join(args)}")
    result = subprocess.run(
        command,
        cwd=_repo_root(),
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 0:
        if result.stdout.strip():
            _status(result.stdout.strip())
        return True

    if result.stdout.strip():
        _status(result.stdout.strip())
    if result.stderr.strip():
        _status(result.stderr.strip())
    return False


def _status(message: str) -> None:
    print(message)
    LOGGER.info(message)


def _relative_to_repo(path: Path) -> str:
    return path.resolve().relative_to(_repo_root()).as_posix()


def _repo_root():
    return Path(__file__).resolve().parents[1]
