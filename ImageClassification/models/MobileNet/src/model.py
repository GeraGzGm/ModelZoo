
import torch
from torch import nn

from ...base_models import ModelsRegistry, BaseModel

@ModelsRegistry.register("MobileNetv1")
class MobileNetV1(BaseModel):
    def __init__(self, n_classes: int):
        super().__init__()
        
        self.features = nn.Sequential(
            ConvBlock(3, 32, 3, 2, 1),     #[112, 112, 32]
            ConvDW(32, 64, 3, 1, 1),       #[112, 112, 64]
            ConvDW(64, 128, 3, 2, 1),      #[56, 56, 128]
            ConvDW(128, 128, 3, 1, 1),     #[56, 56, 128]
            ConvDW(128, 256, 3, 2, 1),     #[28, 28, 256]
            ConvDW(256, 256, 3, 1, 1),     #[28, 28, 256]
            ConvDW(256, 512, 3, 2, 1),     #[14, 14, 512]
            *[ConvDW(512, 512, 3, 1, 1) for _ in range(5)], #[14, 14, 512]
            ConvDW(512, 1024, 3, 2, 1),     #[7, 7, 1024]
            ConvDW(1024, 1024, 3, 1, 1),    #[7, 7, 1024]
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.fc(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
class ConvDW(nn.Module):
    """
    Depthwise Separable convolutions with Depthwise and Pointwise layers followed by BatchNorm and ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)