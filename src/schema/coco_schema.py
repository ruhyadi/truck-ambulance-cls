"""Dataset schema."""

import rootutils

ROOT = rootutils.autosetup()

import json
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel, Field
from tqdm import tqdm

class CocoImageSchema(BaseModel):
    """COCO image schema."""

    id: int = Field(..., example=1)
    file_name: str = Field(..., example="data/images/ppe001.jpg")
    height: int = Field(..., example=720)
    width: int = Field(..., example=1280)


class CocoAnnotationsSchema(BaseModel):
    """COCO annotations schema."""

    id: int = Field(..., example=1)
    image: CocoImageSchema = Field(...)
    category: str = Field(..., example="person")
    bbox: List[Union[int, float]] = Field(..., example=[0.0, 0.0, 100.0, 100.0])


class CocoDatasetSchema(BaseModel):
    """COCO dataset schema."""

    categories: List[str] = Field([])
    images: List[CocoImageSchema] = Field([])
    annotations: List[CocoAnnotationsSchema] = Field([])

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.annotations)

    def load_ppe_json(
        self,
        images_path: Union[str, Path],
        json_path: Union[str, Path],
    ) -> None:
        """Load PPE JSON file."""
        if not isinstance(images_path, Path):
            images_path = Path(images_path)
        json_data = json.load(open(str(json_path), "r"))

        # load categories
        self.categories = [cat["name"] for cat in json_data["categories"]]

        # load images
        for img in json_data["images"]:
            image = CocoImageSchema(
                id=img["id"],
                file_name=str(images_path / img["file_name"]),
                height=img["height"],
                width=img["width"],
            )
            self.images.append(image)

        # load annotations
        self.annotations: List[CocoAnnotationsSchema] = []
        for ann in tqdm(json_data["annotations"], desc="Loading annotations"):
            label = CocoAnnotationsSchema(
                id=ann["id"],
                image=self.images[ann["image_id"] - 1],
                category=self.categories[ann["category_id"] - 1],
                bbox=self.xywh_to_xyxy(ann["bbox"]),
            )
            self.annotations.append(label)

    def xywh_to_xyxy(self, box: List[float]) -> List[int]:
        """Convert xywh to xyxy."""
        x, y, w, h = box

        return [int(x), int(y), int(x + w), int(y + h)]
