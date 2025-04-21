import albumentations as A
from torchvision import datasets
from torch.utils.data import DataLoader

from .utils import Interpolations
from ..base_dataset import DatasetRegistry, BaseDataset, AlbumentationsWrapper

@DatasetRegistry.register("cifar10")
class CIFAR10Dataset(BaseDataset):
    N_CLASSES = 10
    def __init__(self, root: str = "./data"):
        self.root = root

    @classmethod
    def get_number_of_classes(cls) -> int:
        return cls.N_CLASSES
    
    @classmethod
    def input_size(cls):
        return (224, 224)  # Standard size for CNNs

    def get_datasets(self, train_transforms: list[dict[str, str]], test_transforms: list[dict[str, str]]) -> tuple[DataLoader, DataLoader]:
        train_transforms = self.convert_transforms(train_transforms)
        test_transforms = self.convert_transforms(test_transforms)

        train_dataset = AlbumentationsWrapper( datasets.CIFAR10(self.root, train=True, download=True), transform = train_transforms )
        test_dataset = AlbumentationsWrapper( datasets.CIFAR10(self.root, train=False, download=True), transform = test_transforms )
        return train_dataset, test_dataset
    
    def convert_transforms(self, transforms: list[dict[str, str]]) -> A.Compose:
        """
        Converts the transforms given from the json config file into a Compose from Albumentations.

        Args:
            transforms (list[dict[str, str]]): All transforms needed for the given operation.
        Returns:
            A.Compose: Transforms translated to Albumentations.
        """
        augmentations = []

        for transform in transforms:
            type_ = transform.pop("type")

            if interp := transform.get("interpolation"):
                transform["interpolation"] = Interpolations.to_cv2(interp)

            augmentations.append( getattr(A, type_)(**transform) )
        return A.Compose(augmentations)
