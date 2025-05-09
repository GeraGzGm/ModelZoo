from abc import ABC

import torch
from torch import nn

class ModelsRegistry:
    """
    Global registry for DL Architectures.
    
    Usage:
        @ModelsRegistry.register('resnet')
        class ResNet(nn.Module): ...
        
        model = ModelsRegistry.get_model('resnet')
    """
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
        


class BaseModel(nn.Module):

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        Default implementation works for standard models.
        Override this in special architectures (e.g. GoogLeNet).
        """
        return criterion(outputs, targets) 
    

    def get_main_output(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Returns the primary output for metrics calculation.
        Override if your model returns multiple outputs.
        """
        return outputs