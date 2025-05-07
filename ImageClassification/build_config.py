import json
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import Parameters, Optimizers, LossFunctions
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
        datasets, n_classes = self._get_datasets(self.config_file.get("dataset"),
                                                 self.config_file.get("train_transforms"),
                                                 self.config_file.get("inference_transforms"),
                                                 self.config_file.get("train_ratio"))

        model = self._get_model(self.config_file.get("model"), n_classes)
        optimizer = self._get_optimizer( self.config_file.get("optimizer"), model, self.config_file.get("optimizer_kwargs") ),
        loss_function = self._get_criterion( self.config_file.get("loss_function") )

        return Parameters(
            run_type = self.config_file.get("run_type"),
            epochs = self.config_file.get("epochs"),
            batch_size = self.config_file.get("batch_size"),
            val_batch_size = self.config_file.get("val_batch_size"),
            datasets = datasets,
            model = model,
            optimizer = optimizer,
            loss_function = loss_function
        )

    def _get_datasets(self, dataset_name: str, train_transforms: list, inference_transforms: list, split_raio: list[float]) -> tuple[tuple[DataLoader, Optional[DataLoader], DataLoader], int]:
        dataset = DatasetRegistry.get_dataset(dataset_name)()
        return dataset.get_datasets(train_transforms, inference_transforms, split_raio), dataset.get_number_of_classes()
    
    def _get_model(self, model_name: str, n_classes: int) -> nn.Module:
        return ModelsRegistry.get_model(model_name)(n_classes)

    def _get_optimizer(self, optimizer: str, model: nn.Module, kwargs: dict) -> torch.optim.Optimizer:
        return Optimizers.get_optimizer(optimizer)(model.parameters(), **kwargs)

    def _get_criterion(self, loss_function: str) -> nn.Module:
        return LossFunctions.get_criterion(loss_function)()