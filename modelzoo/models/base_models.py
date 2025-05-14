from abc import ABC

import torch
from torch import nn

class ModelsRegistry:
    """
    Global registry for DL Architectures.
    
    Usage:
        @ModelsRegistry.register("resnet", "classification",)
        class ResNet(nn.Module): ...
        
        model, trainer = ModelsRegistry.get_model("resnet")
    """
    _registry = {}
    
    @classmethod
    def register(cls, model_name: str, train_type: str):
        model_name = model_name.lower()
        train_type = train_type.lower()
        def decorator(model_class):            
            cls._registry[model_name] = (model_class, train_type)
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, model_name: str) -> tuple[nn.Module, str]:
        try:
            return cls._registry[model_name.lower()]
        except KeyError:
            raise ValueError(f"Unknown model {model_name}. Available models: {list(cls._registry.keys())}")


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