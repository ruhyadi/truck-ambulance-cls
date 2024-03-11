"""
Ambulance Model Factory.
"""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.mobilenetv3 import MobileNet_V3_Large_Weights

from src.utils.logger import get_logger

log = get_logger()


class AmbulanceMobileNetV3(nn.Module):
    """Ambulance MobileNetV3 model."""

    def __init__(
        self, categories: List[str] = ["ambulance"], pretrained: bool = True
    ) -> None:
        """Initialize the model."""
        super().__init__()

        if pretrained:
            self.backbone = models.mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights
            )
        else:
            self.backbone = models.mobilenet_v3_large()

        # replace the last layer
        num_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(num_features, len(categories))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.backbone(x)
        return x
    
if __name__ == "__main__":
    
    model = AmbulanceMobileNetV3()
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    log.warning(f"Output shape: {y}")