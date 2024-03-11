"""
Ambulance PyTorch Lightning model module.
"""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from torchmetrics import Accuracy, MeanMetric

from src.models.ambulance_model_factory import AmbulanceMobileNetV3
from src.utils.logger import get_logger

log = get_logger()


class AmbulanceLightningModel(LightningModule):
    """Ambulance PyTorch Lightning model."""

    def __init__(
        self,
        backbone: str,
        categories: List[str] = ["ambulance"],
        lr: float = 0.001,
        pretrained: bool = True,
    ) -> None:
        """Initialize the model."""
        super().__init__()

        # save hyperparameters. Call self.hparams.{hyperparameter_name} to access them
        self.save_hyperparameters()

        if backbone == "mobilenetv3":
            self.model = AmbulanceMobileNetV3(
                categories=categories, pretrained=pretrained
            )
        else:
            raise ValueError(f"Backbone {backbone} not supported")

        # loss function single class
        self.criterion = nn.CrossEntropyLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # metrics
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def on_train_start(self) -> None:
        """Actions to perform on train start."""
        log.info(f"Training started")
        self.val_accuracy.reset()
        self.val_loss.reset()

    def step(self, batch: torch.Tensor):
        """Training/validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        y_hat = torch.argmax(y_hat, dim=1)
        y = torch.argmax(y, dim=1)

        return loss, y_hat, y
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, y_hat, y = self.step(batch)
        self.train_loss(loss)
        self.train_accuracy(y_hat, y)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_accuracy, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss, y_hat, y = self.step(batch)
        self.val_loss(loss)
        self.val_accuracy(y_hat, y)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True)
        self.log("val/acc", self.val_accuracy, on_step=True, on_epoch=True)

        return loss
    
    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

if __name__ == "__main__":
    """Debugging."""
    model = AmbulanceLightningModel(backbone="mobilenetv3")
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    log.warning(f"Output shape: {y}")