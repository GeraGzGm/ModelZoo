from .base_models import ModelsRegistry
from .HandwrittenDigitRecognitionWithBackPropNetwork.src.model import LeNet1
from .AlexNet.src.model import AlexNet
from .VGG.src.model import VGG16, VGG19
from .ResNet.src.model import ResNet18


__all__ = ["ModelsRegistry", "LeNet1", "AlexNet", "VGG16", "VGG19", "ResNet18"]