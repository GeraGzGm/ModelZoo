from .base_dataset import DatasetRegistry
from .datasets.cifar10.cifar10 import CIFAR10Dataset
from .datasets.bone_fracture_xray.dataset import BoneFractureXRay

# When importing "from datasets import *" the only things to be imported will be the listed in __all__
__all__ = ["DatasetRegistry",
           "CIFAR10Dataset",
           "BoneFractureXRay"]
