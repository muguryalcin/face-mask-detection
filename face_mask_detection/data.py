from pathlib import Path

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from face_mask_detection.dvc_utils import ensure_data_available

# Supported image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def convert_yolo_box_to_xyxy(
    x_center, y_center, width, height, image_width, image_height
):
    # Convert YOLO format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max) in absolute pixel coordinates.
    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    x_max = (x_center + width / 2) * image_width
    y_max = (y_center + height / 2) * image_height

    # Return the box coordinates
    return (
        max(0.0, min(float(image_width), x_min)),
        max(0.0, min(float(image_height), y_min)),
        max(0.0, min(float(image_width), x_max)),
        max(0.0, min(float(image_height), y_max)),
    )


def collate_detection_batch(batch):
    # Custom collate function to handle variable-length targets for object detection.
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


class FaceMaskDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        split,
        image_size=320,
        min_box_size=1.0,
        class_offset=1,
        normalize=True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = int(image_size)
        self.min_box_size = float(min_box_size)
        # Offset class IDs to reserve 0 for background if needed
        self.class_offset = int(class_offset)
        self.images_dir = self.dataset_dir / split / "images"
        self.labels_dir = self.dataset_dir / split / "labels"
        self.normalize = bool(normalize)
        self.image_paths = self._find_images()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load the image and corresponding labels, preprocess them, and return as tensors.
        image_path = self.image_paths[index]
        image = self._load_image(image_path)
        boxes, labels = self._read_labels(image_path, image.size)
        # Preprocess the image and boxes (resize and filter small boxes)
        # Resize and filter small boxes
        image, boxes = self._resize(image, boxes)
        boxes, labels = self._filter_small_boxes(boxes, labels)

        image = F.to_tensor(image).to(dtype=torch.float32)
        if self.normalize:
            image = F.normalize(
                image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        target = self._target(boxes, labels, index)
        return image, target

    def _find_images(self):
        # Find all image files in the images directory and ensure the corresponding labels directory exists.
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Missing labels directory: {self.labels_dir}")

        # List all image files with supported extensions, sorted for consistency.
        image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        return image_paths

    def _load_image(self, image_path):
        # Load the image using PIL and convert it to RGB format.
        with Image.open(image_path) as image_file:
            return image_file.convert("RGB")

    def _read_labels(self, image_path, image_size):
        # Read the corresponding label file and convert YOLO format labels to absolute pixel coordinates.
        image_width, image_height = image_size
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Warning: Missing label file for {image_path}: {label_path}")
            return [], []

        # Parsing
        boxes = []
        labels = []
        for line_number, line in enumerate(
            label_path.read_text().splitlines(), start=1
        ):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label at {label_path}:{line_number}: {line!r}"
                )

            # Extract info
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:])
            boxes.append(
                convert_yolo_box_to_xyxy(
                    x_center, y_center, width, height, image_width, image_height
                )
            )
            labels.append(class_id + self.class_offset)

        return boxes, labels

    def _resize(self, image, boxes):
        # Resize the image and adjust the bounding boxes accordingly.
        if self.image_size <= 0:
            print("Warning: Image size is not set correctly.")
            return image, boxes

        # Resizing
        original_width, original_height = image.size
        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        boxes = [
            (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            for x1, y1, x2, y2 in boxes
        ]
        return image, boxes

    def _filter_small_boxes(self, boxes, labels):
        # Filter out boxes that are smaller than the minimum box size in either dimension.
        kept_boxes = []
        kept_labels = []
        for box, label in zip(boxes, labels, strict=True):
            x_min, y_min, x_max, y_max = box
            if (x_max - x_min) >= self.min_box_size and (
                y_max - y_min
            ) >= self.min_box_size:
                kept_boxes.append(box)
                kept_labels.append(label)
        return kept_boxes, kept_labels

    def _target(self, boxes, labels, image_id):
        # Convert the list of boxes and labels into tensors and compute additional target information like area and iscrowd for fasterrcnn_resnet50_fpn.
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area.to(dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }


class FaceMaskDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Lazy setup because we need to ensure data is available first
        self.dataset_dir = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # Setting up the datasets
        if self.dataset_dir is None:
            self.dataset_dir = ensure_data_available(self.cfg)

        # Initialize datasets for the appropriate stages
        if stage in (None, "fit"):
            self.train_dataset = self._dataset(self.cfg.data.train_split)
            self.val_dataset = self._dataset(self.cfg.data.val_split)
        if stage in (None, "test", "predict"):
            self.test_dataset = self._dataset(self.cfg.data.test_split)

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, shuffle=False)

    def _dataset(self, split):
        return FaceMaskDetectionDataset(
            self.dataset_dir,
            str(split),
            image_size=self.cfg.preprocessing.image_size,
            min_box_size=self.cfg.preprocessing.min_box_size,
        )

    def _dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=int(self.cfg.training.batch_size),
            shuffle=shuffle,
            num_workers=int(self.cfg.training.num_workers),
            collate_fn=collate_detection_batch,
            pin_memory=torch.cuda.is_available(),
        )
