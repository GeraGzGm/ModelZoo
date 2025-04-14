from abc import ABC, abstractmethod

import numpy as np
from torch import nn
import albumentations as A
from torch.utils.data import Dataset, DataLoader


class ModelsRegistry:
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(model_class):
            cls._registry[name.lower()] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name: str) -> nn.Module:
        try:
            return cls._registry[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown model {name}. Available models: {list(cls._registry.keys())}")
