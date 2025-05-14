import json
from enum import Enum
from typing import Optional, Type

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils import Parameters, Optimizers, LossFunctions, Schedulers
from .models.base_models import ModelsRegistry
from .dataset_loaders.base_dataset import DatasetRegistry
from .train.base_train import TrainRegistry, BaseTraining


class ModelConfigs:
    def __init__(self, config_path: str):
        self.config_file = self.load_json(config_path)

    def load_json(self, path: str) -> dict:
        try:
            with open(path, "r", encoding = "utf-8") as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Given path: {path} does not exist.") from e
        
    def get_model_configs(self) -> Parameters:
        datasets, classes = self._get_datasets(self.config_file.get("dataset"),
                                                 self.config_file.get("train_transforms"),
                                                 self.config_file.get("inference_transforms"),
                                                 self.config_file.get("batch_size"),
                                                 self.config_file.get("inference_batch_size"),
                                                 self.config_file.get("train_ratio"))

        model_kwargs = self.config_file.get("model_kwargs", {})
        model, train_type = self._get_model(self.config_file.get("model"), len(classes), model_kwargs)

        optimizer = self._get_optimizer( self.config_file.get("optimizer"), model, self.config_file.get("optimizer_kwargs") )
        loss_function = self._get_criterion( self.config_file.get("loss_function") )
        scheduler = self._get_scheduler(optimizer, self.config_file.get("scheduler") , self.config_file.get("scheduler_kwargs")  )

        return Parameters(
            epochs = self.config_file.get("epochs"),
            batch_size = self.config_file.get("batch_size"),
            val_batch_size = self.config_file.get("val_batch_size"),
            model = model,
            optimizer = optimizer,
            loss_function = loss_function,
            labels = classes,
            datasets = datasets,
            inferece_transforms = self.config_file.get("inference_transforms"),
            scheduler = scheduler,
            train_type = train_type
        )

    def _get_datasets(self, dataset_name: str,
                      train_transforms: list,
                      inference_transforms: list,
                      train_batch_size: int,
                      inference_batch_size: int,
                      split_raio: list[float]) -> tuple[tuple[DataLoader, Optional[DataLoader], DataLoader], Enum]:
        dataset = DatasetRegistry.get_dataset(dataset_name)(train_batch_size, inference_batch_size)
        return dataset.get_datasets(train_transforms, inference_transforms, split_raio), dataset.get_classes()
    
    def _get_model(self, model_name: str, n_classes: int, model_kwargs: dict | None) -> tuple[nn.Module, str]:
        """
        Retrieve model from the ModelsRegistry and the training type.
        """
        model, train_type = ModelsRegistry.get_model(model_name)
        return model(n_classes, **model_kwargs), train_type

    @staticmethod
    def get_trainer(train_type: str) -> Type[BaseTraining]:
        return TrainRegistry.get_model(train_type)

    def _get_optimizer(self, optimizer: str, model: nn.Module, kwargs: dict) -> torch.optim.Optimizer:
        """
        Retrieve optimizer from the Optimizers dataclass.
        """
        return Optimizers.get_optimizer(optimizer)(model.parameters(), **kwargs)

    def _get_criterion(self, loss_function: str) -> nn.Module:
        """
        Retrieve loss function from the LossFunctions dataclass.
        """
        return LossFunctions.get_criterion(loss_function)()
    
    def _get_scheduler(self, optim: torch.optim.Optimizer, scheduler: dict, scheduler_kwargs: dict) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """
        Retrieve loss function from the LossFunctions dataclass.
        """
        if not scheduler:
            return None
        return Schedulers.get_scheduler(scheduler)(optim, **scheduler_kwargs)