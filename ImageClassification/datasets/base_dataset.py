from abc import ABC, abstractmethod

import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader

class BaseDataset(ABC):
    """Abstract base class for all datasets"""
    @classmethod
    @abstractmethod
    def input_size(cls) -> tuple[int, int]:
        """Return (height, width) for the model"""
        pass

    @abstractmethod
    def get_datasets(self, transforms: dict[str, str]) -> tuple[Dataset, Dataset]:
        """Return (train_dataset, val_dataset)"""
        pass

    @abstractmethod
    def get_number_of_classes(self) -> int:
        """Return the number of classes. This will work for the FC layers output."""
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
        
    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return image, label