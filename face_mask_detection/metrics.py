import torch
from lightning.pytorch.callbacks import Callback
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


class MetricsHistory(Callback):
    # Custom callback to store metrics
    def __init__(self):
        self.history = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._collect(trainer.logged_metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._collect(trainer.callback_metrics)

    def _collect(self, metrics):
        for name, value in metrics.items():
            scalar = to_scalar(value)
            if scalar is not None:
                self.history.setdefault(str(name), []).append(scalar)


class DetectionMetrics:
    def __init__(self, class_names, score_threshold=0.5):
        self.class_names = list(class_names)
        self.class_ids = range(1, len(self.class_names))
        self.score_threshold = float(score_threshold)
        self.map_50 = MeanAveragePrecision(iou_thresholds=[0.5], class_metrics=True)
        self.map_95 = MeanAveragePrecision(iou_thresholds=[0.95], class_metrics=True)
        self.predictions = []
        self.targets = []

    def reset(self):
        # Reset the internal state
        self.map_50.reset()
        self.map_95.reset()
        self.predictions = []
        self.targets = []

    def update(self, predictions, targets):
        # Get predictions and targets
        predictions = [
            {
                "boxes": prediction["boxes"].detach().cpu(),
                "labels": prediction["labels"].detach().cpu(),
                "scores": prediction["scores"].detach().cpu(),
            }
            for prediction in predictions
        ]
        targets = [
            {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }
            for target in targets
        ]

        # Update the internal state since we compute them over the whole set not just the batch
        self.map_50.update(predictions, targets)
        self.map_95.update(predictions, targets)
        self.predictions.extend(predictions)
        self.targets.extend(targets)

    def compute(self):
        # Compute the metrics
        map_50 = self.map_50.compute()
        map_95 = self.map_95.compute()
        precision, recall = precision_recall(
            self.predictions,
            self.targets,
            self.class_ids,
            iou_threshold=0.5,
            score_threshold=self.score_threshold,
        )

        # Format
        metrics = {
            "precision": precision,
            "recall": recall,
            "map_50": _metric_value(map_50["map"]),
            "map_95": _metric_value(map_95["map"]),
        }
        metrics.update(self._per_class_ap(map_50, "ap_50"))
        metrics.update(self._per_class_ap(map_95, "ap_95"))
        return metrics

    def _per_class_ap(self, result, prefix):
        metrics = {
            f"{prefix}_{self._class_name(class_id)}": 0.0 for class_id in self.class_ids
        }
        for class_id, ap in zip(
            result["classes"], result["map_per_class"], strict=True
        ):
            class_id = int(class_id)
            if class_id in self.class_ids:
                metrics[f"{prefix}_{self._class_name(class_id)}"] = _metric_value(ap)
        return metrics

    def _class_name(self, class_id):
        return str(self.class_names[class_id]).replace(" ", "_")


def precision_recall(predictions, targets, class_ids, iou_threshold, score_threshold):
    # Calculate the precision and recall for a given iou_threshold
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for prediction, target in zip(predictions, targets, strict=True):
        for class_id in class_ids:
            pred_boxes = _boxes_for_class(prediction, class_id, score_threshold)
            target_boxes = target["boxes"][target["labels"] == class_id]
            matched_targets = set()

            ious = (
                box_iou(pred_boxes, target_boxes)
                if len(pred_boxes) and len(target_boxes)
                else None
            )

            for pred_index in range(len(pred_boxes)):
                if ious is None:
                    false_positives += 1
                    continue

                best_iou, best_target = torch.max(ious[pred_index], dim=0)
                best_target = int(best_target)
                if (
                    float(best_iou) >= iou_threshold
                    and best_target not in matched_targets
                ):
                    true_positives += 1
                    matched_targets.add(best_target)
                else:
                    false_positives += 1

            false_negatives += len(target_boxes) - len(matched_targets)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return precision, recall


def to_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _boxes_for_class(prediction, class_id, score_threshold):
    # Filter the predicted boxes for a specific class and score threshold.
    keep = (prediction["labels"] == class_id) & (
        prediction["scores"] >= score_threshold
    )
    return prediction["boxes"][keep]


def _metric_value(value):
    value = to_scalar(value)
    if value is None or value < 0:
        return 0.0
    return value
