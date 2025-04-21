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

    def get_datasets(self, transforms: dict[str, A.Compose]) -> tuple[DataLoader, DataLoader]:
        transforms = self.convert_transforms(transforms)

        train_dataset = AlbumentationsWrapper( datasets.CIFAR10(self.root, train=True, download=True), transform = transforms )
        test_dataset = AlbumentationsWrapper( datasets.CIFAR10(self.root, train=False, download=True), transform = transforms )
        return train_dataset, test_dataset
    
    def convert_transforms(self, transforms: list[dict[str, str]]) -> A.Compose:
        augmentations = []

        for transform in transforms:
            type_ = transform.pop("type")

            if interp := transform.get("interpolation"):
                transform["interpolation"] = Interpolations.to_cv2(interp)

            augmentations.append( getattr(A, type_)(**transform) )

        return A.Compose(augmentations)
