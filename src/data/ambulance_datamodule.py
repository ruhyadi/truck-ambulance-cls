"""Ambulance lightning datamodule."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.ambulance_dataset import AmbulanceDataset


class AmbulanceDataModule(LightningDataModule):
    """Ambulance lightning datamodule."""

    def __init__(
        self,
        images_path: str,
        train_json_path: str,
        val_json_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        """Initialize Ambulance lightning datamodule."""
        super().__init__()
        self.images_path = images_path
        self.train_json_path = train_json_path
        self.val_json_path = val_json_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets."""
        if not self.data_train and not self.data_val:
            self.data_train = AmbulanceDataset(
                images_path=self.images_path, json_path=self.train_json_path
            )
            self.data_val = AmbulanceDataset(
                images_path=self.images_path, json_path=self.val_json_path
            )

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader."""
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
