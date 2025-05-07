from typing import Optional
from enum import Enum, auto

import torch
import albumentations as A
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

from .labels import Labels
from ..utils import Interpolations
from ...base_dataset import DatasetRegistry, BaseDataset, AlbumentationsWrapper

torch.manual_seed(0)

@DatasetRegistry.register("cifar10")
class CIFAR10Dataset(BaseDataset):
    LABELS = Labels
    NUM_WORKERS = 4
    def __init__(self, train_batch_size: int, inference_batch_size: int, root: str = "./data"):
        self.root = root
        self.train_batch_size = train_batch_size
        self.inference_batch_size = inference_batch_size

    @classmethod
    def get_classes(cls) -> Enum:
        return cls.LABELS
    
    @classmethod
    def input_size(cls):
        return (224, 224)  # Standard size for CNNs

    def get_datasets(self, train_transforms: list[dict[str, str]], test_transforms: list[dict[str, str]], split_ratio: list[float]) -> tuple[DataLoader, Optional[DataLoader], DataLoader]:
        train_transforms = self.convert_transforms(train_transforms)
        test_transforms = self.convert_transforms(test_transforms)

        dataset = AlbumentationsWrapper( datasets.CIFAR10(self.root, train = True, download = True), transform = train_transforms )
        train_dataset, val_dataset = self._split_train(dataset, split_ratio)

        test_dataset = AlbumentationsWrapper( datasets.CIFAR10(self.root, train = False, download = True), transform = test_transforms )
        return self._create_dataloaders(train_dataset, val_dataset, test_dataset)
    
    def convert_transforms(self, transforms: list[dict[str, str]]) -> A.Compose:
        """
        Converts the transforms given from the json config file into a Compose from Albumentations.

        Args:
            transforms (list[dict[str, str]]): All transforms needed for the given operation.
        Returns:
            A.Compose: Transforms translated to Albumentations.
        """
        augmentations = []

        for transform in transforms:
            type_ = transform.pop("type")

            if interp := transform.get("interpolation", None):
                transform["interpolation"] = Interpolations.to_cv2(interp)

            augmentations.append( getattr(A, type_)(**transform) )
        return A.Compose(augmentations + [A.ToTensorV2()])

    def _split_train(self, dataset: AlbumentationsWrapper, split_ratio: list[float]) -> tuple[AlbumentationsWrapper, AlbumentationsWrapper]:
        return random_split(dataset, split_ratio)
    
    def _create_dataloaders(self, trainset: AlbumentationsWrapper, valset : AlbumentationsWrapper, testset: AlbumentationsWrapper):
        return (
            DataLoader(trainset, batch_size = self.train_batch_size, shuffle = True, num_workers = self.NUM_WORKERS),
            DataLoader(valset, batch_size = self.inference_batch_size, shuffle = True, num_workers = self.NUM_WORKERS),
            DataLoader(testset, batch_size = self.inference_batch_size, shuffle = True, num_workers = self.NUM_WORKERS),
        )
