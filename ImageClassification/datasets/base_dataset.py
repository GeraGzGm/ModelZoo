import os
import logging
from enum import Enum, EnumMeta, Flag, EnumType
from abc import ABC, abstractmethod, ABCMeta

import numpy as np
from torch import from_numpy, Tensor
from torch.utils.data import Dataset, DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi

logger = logging.getLogger(__name__)

class BaseDataset(ABC):
    """Abstract base class for all datasets"""
    @classmethod
    @abstractmethod
    def input_size(cls) -> tuple[int, int]:
        """Return (height, width) for the model"""
        pass

    @abstractmethod
    def get_datasets(self, train_transforms: dict[str, str], test_transforms: dict[str, str]) -> tuple[Dataset, Dataset]:
        """Return (train_dataset, val_dataset)"""
        pass

    @abstractmethod
    def get_classes(self) -> Enum:
        """Return the classes of the dataset."""
        pass

class KaggleDataset:
    """
    Base class for the Kaggle Datasets.

    Args:
        dataset_name: Name of the user and dataset.
        dataset_path: Path where the dataset will be stored.
    
    Example:
        @DatasetRegistry.register("braincancer-mri")
        class BrainCancer_MRI(KaggleDataset):
            def __init__(self).
                super().__init__("orvile/brain-cancer-mri-dataset", "./ImageClassification/braincaner-mri/")
    
    """
    def __init__(self, dataset_name: str, dataset_path: str):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        self.init_kaggle()
        self.download_dataset()

    def init_kaggle(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def download_dataset(self) -> None:
        if not self._is_dataset_already_created():
            logger.info(f"Downloading dataset {self.dataset_name} in: {self.dataset_path}")
            self.api.dataset_download_files(self.dataset_name, path = self.dataset_path, unzip=True)

        logger.info(f"Dataset {self.dataset_name} already downloaded at: {self.dataset_path}")

    def _is_dataset_already_created(self) -> bool:
        """
        Check if the path already exists and if the path has files in order to define if the Dataset is already downloaded.

        Returns:
            bool: False if the dataset needs to be downloaded or True if it already exists.
        """
        if not os.path.exists(self.dataset_path):
            return False
        
        if not any(os.scandir(self.dataset_path)):
            return False
        return True
    


class ABCEnumMeta(ABCMeta, EnumType):
    pass

class BaseLabels(ABC, Enum, metaclass = ABCEnumMeta):
    """Abstract class for dataset-specific label enums."""

    @classmethod
    @abstractmethod
    def get_key(cls, value: int) -> str:
        """Returns the name of the label corresponding to the index."""
        pass
    
    @classmethod
    @abstractmethod
    def __len__(self):
        """Returns the number of labels."""
        pass

class DatasetRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(dataset_class):
            cls._registry[name.lower()] = dataset_class
            return dataset_class
        return decorator
    
    @classmethod
    def get_dataset(cls, name: str) -> BaseDataset:
        try:
            return cls._registry[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown dataset {name}. Available: {list(cls._registry.keys())}")

class AlbumentationsWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx) -> tuple[np.ndarray, Tensor]:
        image, label = self.dataset[idx]
        image = np.array(image)
        label = np.array(label)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return image, from_numpy(label)