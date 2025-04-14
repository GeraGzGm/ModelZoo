import cv2
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

class DataLoaderCIFAR10:
    ROOT = "./Datasets/"
    def __init__(self, batch_size: int, test_batch_size: int = 4, num_workers: int = 8):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_batch_size = test_batch_size

        # CIFAR-100 stats
        self.mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
        self.std =  [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

        self.train_transforms = A.Compose([
            A.Resize(256, 256, interpolation = cv2.INTER_LINEAR, p = 1), # To match the size they use at the paper.
            A.HorizontalFlip(0.5),
            A.RandomCrop(224, 224, p = 1),
            A.Normalize(self.mean, self.std, p = 1), # Instead of the PCA method they use in the paper.
            A.ToTensorV2()
        ])

        self.test_transforms = A.Compose([
            A.Resize(224, 224, interpolation = cv2.INTER_LINEAR, p = 1),
            A.Normalize(self.mean, self.std, p = 1), # Instead of the PCA method they use in the paper.
            A.ToTensorV2()
        ])

    def get_loaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset, test_dataset = self._download_datasets()
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers, pin_memory = True)
        test_loader = DataLoader(test_dataset, batch_size = self.test_batch_size, shuffle = False, num_workers = self.num_workers, pin_memory = True)
        return train_loader, test_loader
    
    def _download_datasets(self) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
        train_dataset = AlbumentationsCIFAR10(self.ROOT, transform = self.train_transforms, train = True, download = True)
        test_dataset = AlbumentationsCIFAR10(self.ROOT, transform = self.test_transforms, train = False, download = True)
        return train_dataset, test_dataset
    
class AlbumentationsCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.dataset = datasets.CIFAR10(root = root, train = train, download = download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label