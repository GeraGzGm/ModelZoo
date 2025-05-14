import os
import subprocess
from typing import Type, Optional
from abc import ABC, abstractmethod

import torch
from torch import nn

from ..utils import Parameters

class BaseTraining(ABC):
    """
    Base class for training models.

    Args:
        model: The model to train.
        optimizer: The optimizer for updating model weights.
        criterion: The loss function.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run the training on (e.g., "cpu" or "cuda").
    """
    BOARD_PORT = "6006"

    def __init__(self, config: Parameters, out_dir: Optional[str] = None, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model = config.model
        self.optimizer = config.optimizer
        self.criterion = config.loss_function
        self.scheduler = config.scheduler

        self.config = config

        self.trainset = config.datasets[0]
        self.valset = config.datasets[1]
        self.testset = config.datasets[2]
        self.out_dir = out_dir
        self.model_path = model_path

        self._move_to_device()

    def _move_to_device(self) -> None:
        if torch.cuda.is_available() and self.device == "cuda":
            self.model.to(self.device)
            self.criterion.to(self.device)

    def _init_tensorboard(self, out_dir: str):
        subprocess.Popen(["tensorboard", f"--logdir={out_dir}", "--port", self.BOARD_PORT])

    def step_scheduler(self, val_loss: float) -> None:
        if self.scheduler:
            self.scheduler.step(val_loss)
    
    def _save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok = True)
        torch.save(self.model.state_dict(), path)
    
    def _load_model(self, load_path: str | None) -> None:
        if load_path:
            try:
                state_dict = torch.load(load_path, map_location = self.device)
                self.model.load_state_dict(state_dict)
            except TypeError as e:
                raise "Error loading the model." from e

    def __call__(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def _train_epoch(self):
        pass

    @abstractmethod
    def _tensorboard_log(self) -> None:
        pass

class TrainRegistry:
    """
    Global registry for different kind of trainings.
    
    Usage:
        @TrainRegistry.register("classification",)
        class BaseClassification: ...
        
        trainer = ModelsRegistry.get_model("classification")
    """
    _registry = {}
    
    @classmethod
    def register(cls, train_type: str):
        def decorator(train_class):            
            cls._registry[train_type.lower()] = train_class
            return train_class
        return decorator
    
    @classmethod
    def get_model(cls, train_type: str) -> Type[BaseTraining]:
        try:
            return cls._registry[train_type.lower()]
        except KeyError:
            raise ValueError(f"Unknown train type: {train_type}. Available types: {list(cls._registry.keys())}")



