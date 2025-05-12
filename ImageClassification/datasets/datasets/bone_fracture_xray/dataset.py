import os
from typing import Optional
from enum import Enum, auto

import torch
import albumentations as A
from torchvision import datasets
from kaggle.api.kaggle_api_extended import KaggleApi
from torch.utils.data import DataLoader, random_split, Dataset

from .labels import Labels
from ..utils import Interpolations
from ...base_dataset import DatasetRegistry, KaggleDataset, BaseDataset

@DatasetRegistry.register("BoneFracture-XRay")
class BoneFractureXRay(KaggleDataset, BaseDataset):
    LABELS = Labels
    DATASET_PATH = "./data/bonefracture/"
    KAGGLE_DATASET = "bmadushanirodrigo/fracture-multi-region-x-ray-data"
    
    def __init__(self, train_batch_size: int, inference_batch_size: int, *args):
        super().__init__(self.KAGGLE_DATASET, self.DATASET_PATH)

    def get_datasets(self, train_transforms: list[dict[str, str]], test_transforms: list[dict[str, str]], split_ratio: list[float]) -> tuple[DataLoader, Optional[DataLoader], DataLoader]:
        return None
    
    @classmethod
    def get_classes(cls) -> Enum:
        return cls.LABELS
    
    @classmethod
    def input_size(cls):
        return (224, 224)  # Standard size for CNNs

if __name__ == "__main__":
    KaggleDataset()