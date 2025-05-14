import torch
import pytest

from train.base_train import TrainRegistry, BaseTraining
from models.base_models import ModelsRegistry, BaseModel
from dataset_loaders.base_dataset import DatasetRegistry, BaseDataset

class DummyTraining(BaseTraining):
    def train(self): pass
    def _train_epoch(self): pass

class DummyModel(BaseModel):
    pass

class DummyDataset(BaseDataset):
    @classmethod
    def input_size(cls) -> tuple[int, int]:
        pass

    def get_datasets(self):
        pass

    def get_classes(self):
        pass

def test_train_registry_register_and_get():
    TrainRegistry.register("dummy_train")(DummyTraining)
    trainer_cls = TrainRegistry.get_model("dummy_train")

    assert trainer_cls is DummyTraining

def test_train_registry_unknown_type():
    with pytest.raises(ValueError):
        TrainRegistry.get_model("unknown_type")

def test_models_registry_register_and_get():
    ModelsRegistry.register("dummy_model", "dummy_train")(DummyModel)
    model_cls, train_type = ModelsRegistry.get_model("dummy_model")

    assert model_cls is DummyModel
    assert train_type == "dummy_train"

def test_models_registry_unknown_model():
    with pytest.raises(ValueError):
        ModelsRegistry.get_model("unknown_model")

def test_datasets_registry_register_and_get():
    DatasetRegistry.register("dummy_dataset")(DummyDataset)
    dataset_cls = DatasetRegistry.get_dataset("dummy_dataset")

    assert dataset_cls is DummyDataset

def test_datasets_registry_unknown_dataset():
    with pytest.raises(ValueError):
        DatasetRegistry.get_dataset("unknown_dataset")