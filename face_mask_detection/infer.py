import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F

from face_mask_detection.data import NORMALIZE_MEAN, NORMALIZE_STD
from face_mask_detection.dvc_utils import ensure_data_available
from face_mask_detection.model import FaceMaskDetector


def infer(cfg, image_path, checkpoint_path=None):
    # Ensure the data is available and load the model
    ensure_data_available(cfg)
    model = _load_model(cfg, checkpoint_path)
    model.eval()

    # Load the image and preprocess it
    image = _load_image_tensor(image_path, int(cfg.preprocessing.image_size))
    device = next(model.parameters()).device
    image = image.to(device)

    # Get outputs
    with torch.no_grad():
        output = model([image])[0]

    # Format and print detections
    detections = _format_detections(cfg, output)
    if not detections:
        print("No detections above threshold")
        return detections

    # Print detections in a readable format
    for detection in detections:
        box = detection["box"]
        print(
            f"{detection['class']} "
            f"score={detection['score']:.4f} "
            f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
        )
    return detections


def _load_model(cfg, checkpoint_path):
    # Loads the model
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return FaceMaskDetector.load_from_checkpoint(str(checkpoint), cfg=cfg)
    return FaceMaskDetector(cfg)


def _load_image_tensor(image_path, image_size, normalize):
    # Load the image, resize it, convert to tensor, and optionally normalize it.
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Load, convert to RGB, resize, return as tensor
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    values = torch.tensor(list(image.getdata()), dtype=torch.float32)
    tensor = values.reshape(image_size, image_size, 3).permute(2, 0, 1) / 255.0
    if normalize:
        tensor = F.normalize(tensor, mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    return tensor


def _format_detections(cfg, output):
    # Formats the raw model output into a list of detections with class names, scores, and bounding boxes.j
    threshold = float(cfg.inference.score_threshold)
    max_detections = int(cfg.inference.max_detections)
    class_names = list(cfg.data.class_names)

    detections = []
    for box, label, score in zip(
        output.get("boxes", []),
        output.get("labels", []),
        output.get("scores", []),
    ):
        # Filter detections based on the score threshold
        score_value = float(score.item())
        if score_value < threshold:
            continue

        label_value = int(label.item())
        if label_value < 0 or label_value >= len(class_names):
            logging.warning("Invalid class ID %s in output, skipping", label_value)
            continue
        class_name = class_names[label_value]
        detections.append(
            {
                "class": class_name,
                "label": label_value,
                "score": score_value,
                "box": [float(value) for value in box.detach().cpu().tolist()],
            }
        )
        # Limit the number of detections to max_detections
        if len(detections) >= max_detections:
            break

    return detections
