from .base_models import ModelsRegistry
from .HandwrittenDigitRecognitionWithBackPropNetwork.src.model import LeNet1
from .AlexNet.src.model import AlexNet
from .VGG.src.model import VGG16, VGG19
from .ResNet.src.model import ResNet18
from .GoogLeNet.src.model import InceptionV1
from .MobileNet.v1.src.model import MobileNetV1
from .MobileNet.v2.src.model import MobileNetV2


__all__ = ["ModelsRegistry",
           "LeNet1",
           "AlexNet",
           "VGG16",
           "VGG19",
           "ResNet18",
           "InceptionV1",
           "MobileNetv1",
           "MobileNetV2"
           ]