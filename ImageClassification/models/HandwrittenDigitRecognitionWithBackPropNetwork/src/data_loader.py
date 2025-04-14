from typing import Optional

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class MNISTDataLoader:
    def __init__(self, transform: Optional[transforms.Compose], batch_size: int = 64, data_dir: str = "./Datasets/"):
        self.transform = transform
        self.batch_size = batch_size
        self.data_dir = data_dir

    def get_loaders(self):
        train_dataset, test_dataset = self.get_datasets(self.transform)

        train_loader = DataLoader( train_dataset, batch_size = self.batch_size, shuffle = True,  num_workers = 2)
        test_loader = DataLoader( test_dataset, batch_size = self.batch_size, shuffle = False,  num_workers = 2)

        return train_loader, test_loader
    
    def get_datasets(self, transform) -> tuple[Dataset]:
        train_dataset = datasets.MNIST( root = './Datasets/', train = True, download = True, transform = transform)
        test_dataset = datasets.MNIST( root = './Datasets/', train = False, download = True, transform = transform)
        return train_dataset, test_dataset