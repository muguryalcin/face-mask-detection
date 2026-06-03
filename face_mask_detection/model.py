import torch
from lightning import LightningModule
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FaceMaskDetector(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.detector = self._build_detector()

    def forward(self, images):
        return self.detector(images)

    def training_step(self, batch, batch_idx):
        # Forward pass
        images, targets = batch
        loss_dict = self.detector(images, targets)
        loss = self._sum_losses(loss_dict)
        self._log_losses(
            "train", loss, loss_dict, len(images), on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # One forward pass to compute validation losses without backpropagation
        images, targets = batch
        loss_dict = self._validation_loss_dict(images, targets)
        loss = self._sum_losses(loss_dict)
        self._log_losses(
            "val", loss, loss_dict, len(images), on_step=False, on_epoch=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.model.learning_rate),
            weight_decay=float(self.cfg.model.weight_decay),
        )

    def _build_detector(self):
        if self.cfg.model.name != "fasterrcnn_resnet50_fpn":
            raise ValueError("Only fasterrcnn_resnet50_fpn is supported")

        weights = None
        if bool(self.cfg.model.pretrained):
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

        detector = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            # Freeze backbone layers if trainable_backbone_layers is 0, otherwise unfreeze the last N layers as specified by trainable_backbone_layers.
            trainable_backbone_layers=int(self.cfg.model.trainable_backbone_layers),
        )
        # Replace the default box predictor with a new one that has the correct number of output classes for our dataset.
        in_features = detector.roi_heads.box_predictor.cls_score.in_features
        detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            int(self.cfg.data.num_classes),
        )
        return detector

    def _validation_loss_dict(self, images, targets):
        # Compute the loss for validation without backprop
        was_training = self.detector.training
        self.detector.train()
        with torch.no_grad():
            loss_dict = self.detector(images, targets)
        if not was_training:
            self.detector.eval()
        return loss_dict

    def _sum_losses(self, loss_dict):
        # Adds all of the losses
        return sum(loss for loss in loss_dict.values())

    def _log_losses(self, prefix, loss, loss_dict, batch_size, on_step, on_epoch):
        # Logging the losses.
        self.log(
            f"{prefix}/loss",
            loss,
            prog_bar=True,
            on_step=on_step,
            on_epoch=on_epoch,
            batch_size=batch_size,
        )
        for name, value in loss_dict.items():
            self.log(
                f"{prefix}/{name}",
                value,
                prog_bar=False,
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=batch_size,
            )
