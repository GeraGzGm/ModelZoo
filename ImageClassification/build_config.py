import json
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop, Adamax

from utils import Parameters, Optimizers
from models.base_models import ModelsRegistry
from datasets.base_dataset import DatasetRegistry


class ModelConfigs:
    def __init__(self, config_path: str):
        self.config_file = self.load_json(config_path)

    def load_json(self, path: str) -> dict:
        try:
            with open(path, "r", encoding = "utf-8") as file:
                return json.load(file)
        except FileNotFoundError as e:
            assert e(f"Given path: {path} does not exist.")
        
    def get_model_configs(self) -> Parameters:
        datasets = self._get_datasets(self.config_file.get("dataset"), self.config_file.get("transforms")),

        # TODO: CHANGE THE 10 to the number of classes of the dataset

        model = self._get_model(self.config_file.get("model"), 10 )
        optimizer = self._get_optimizer( self.config_file.get("optimizer"), model, self.config_file.get("optimizer_kwargs") ),

        return Parameters(
            epochs = self.config_file.get("epochs"),
            batch_size = self.config_file.get("batch_size"),
            val_batch_size = self.config_file.get("val_batch_size"),
            datasets = datasets,
            model = model,
            optimizer = optimizer
        )

    def _get_datasets(self, dataset_name: str, transforms: list) -> tuple[DataLoader, DataLoader]:
        dataset = DatasetRegistry.get_dataset(dataset_name)()
        return dataset.get_datasets(transforms)
    
    def _get_model(self, model_name: str, n_classes: int) -> nn.Module:
        return ModelsRegistry.get_model(model_name)(n_classes)

    def _get_optimizer(self, optimizer: str, model: nn.Module, kwargs: dict) -> torch.optim.Optimizer:
        return Optimizers.get_optimizer(optimizer)(model.parameters(), **kwargs)
