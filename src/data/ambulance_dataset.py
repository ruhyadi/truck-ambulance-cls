"""Ambulance pytorch datamodule."""

import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path
from typing import List, Tuple

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.schema.coco_schema import CocoDatasetSchema
from src.utils.logger import get_logger

log = get_logger()


class AmbulanceDataset(Dataset):
    """Ambulance PyTorch dataset."""

    def __init__(self, images_path: str, json_path: str) -> None:
        """Initialize Ambulance PyTorch dataset."""
        self.images_path = Path(images_path)
        self.json_path = Path(json_path)

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.coco = self.load_coco()

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.coco)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, np.ndarray]:
        """Return sample from dataset."""
        label = self.coco.annotations[index]
        image = self.crop_image(img_path=label.image.file_name, bbox_xyxy=label.bbox)
        image: torch.Tensor = self.transform(image)

        # label to one-hot encoding
        target = np.zeros(len(self.coco.categories))
        target[self.coco.categories.index(label.category)] = 1

        return image, target

    def load_coco(self) -> CocoDatasetSchema:
        """Load COCO dataset."""
        coco = CocoDatasetSchema()
        coco.load_ppe_json(
            images_path=self.images_path,
            json_path=self.json_path,
        )
        log.info(f"Loaded {len(coco)} annotations")

        return coco

    def crop_image(self, img_path: str, bbox_xyxy: List[int]) -> np.ndarray:
        """Crop image."""
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[
            bbox_xyxy[1] : bbox_xyxy[3],
            bbox_xyxy[0] : bbox_xyxy[2],
        ]

        return image
