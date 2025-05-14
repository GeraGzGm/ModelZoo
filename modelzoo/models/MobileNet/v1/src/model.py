from math import floor

import torch
from torch import nn

from ....base_models import ModelsRegistry, BaseModel

@ModelsRegistry.register("MobileNetv1", "Classification")
class MobileNetV1(BaseModel):
    """
    Implementation of MobileNetV1.

    Args:
        n_classes (int): Number of classes for the output of the FC layer.
        alpha (float): Width multiplier -> (0,1]
        epsilon (float): Resolution multiplier -> (0,1]
    """
    def __init__(self, n_classes: int, **kwargs):
        super().__init__()

        self.alpha = kwargs.get("alpha", 1)
        #TODO: Implement epsilon ( but this must be done at the Dataset loader)
        self.epsilon = kwargs.get("epsilon", 1)
        
        self.features = nn.Sequential(
            ConvBlock(3, floor(32*self.alpha), 3, 2, 1),                       #[112, 112, 32]
            ConvDW(floor(32*self.alpha), floor(64*self.alpha), 3, 1, 1),       #[112, 112, 64]
            ConvDW(floor(64*self.alpha), floor(128*self.alpha), 3, 2, 1),      #[56, 56, 128]
            ConvDW(floor(128*self.alpha), floor(128*self.alpha), 3, 1, 1),     #[56, 56, 128]
            ConvDW(floor(128*self.alpha), floor(256*self.alpha), 3, 2, 1),     #[28, 28, 256]
            ConvDW(floor(256*self.alpha), floor(256*self.alpha), 3, 1, 1),     #[28, 28, 256]
            ConvDW(floor(256*self.alpha), floor(512*self.alpha), 3, 2, 1),     #[14, 14, 512]
            *[ConvDW(floor(512*self.alpha), floor(512*self.alpha), 3, 1, 1) for _ in range(5)], #[14, 14, 512]
            ConvDW(floor(512*self.alpha), floor(1024*self.alpha), 3, 2, 1),     #[7, 7, 1024]
            ConvDW(floor(1024*self.alpha), floor(1024*self.alpha), 3, 1, 1),    #[7, 7, 1024]
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(floor(1024*self.alpha), n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc( self.features(x) )


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