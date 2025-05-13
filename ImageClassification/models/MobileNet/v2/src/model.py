from math import floor

import torch
from torch import nn

from ....base_models import ModelsRegistry, BaseModel

@ModelsRegistry.register("MobileNetv2")
class MobileNetV2(BaseModel):
    """
    Implementation of MobileNetV2.

    Args:
        n_classes (int): Number of classes for the output of the FC layer.
        alpha (float): Width multiplier -> (0,1]
    """
    def __init__(self, n_classes: int, **kwargs):
        super().__init__()
        self.alpha = kwargs.get("alpha", 1)

        c = lambda w: max(1, int(w * self.alpha)) # Width adjustment function
            
        self.features = nn.Sequential(
            # Initial conv (224x224x3 → 112x112x32)
            ConvBlock(3, c(32), 3, 2, 1),  
            
            # Bottleneck layers (t, c, n, s)
            # t=1, c=16, n=1, s=1
            BottleNeckBlock(c(32), c(16), 3, stride=1, padding = 1, expansion_ratio=1),  # No expansion
            
            # t=6, c=24, n=2, s=2 (then s=1)
            BottleNeckBlock(c(16), c(24), 3, stride=2, padding = 1, expansion_ratio=6),
            BottleNeckBlock(c(24), c(24), 3, stride=1, padding = 1, expansion_ratio=6),
            
            # t=6, c=32, n=3, s=2 (then s=1)
            BottleNeckBlock(c(24), c(32), 3, stride=2, padding = 1, expansion_ratio=6),
            *[BottleNeckBlock(c(32), c(32), 3, stride=1, padding = 1, expansion_ratio=6) for _ in range(2)],
            
            # t=6, c=64, n=4, s=2 (then s=1)
            BottleNeckBlock(c(32), c(64), 3, stride=2, padding = 1, expansion_ratio=6),
            *[BottleNeckBlock(c(64), c(64), 3, stride=1, padding = 1, expansion_ratio=6) for _ in range(3)],
            
            # t=6, c=96, n=3, s=1
            BottleNeckBlock(c(64), c(96), 3, stride=1, padding = 1, expansion_ratio=6),
            *[BottleNeckBlock(c(96), c(96), 3, stride=1, padding = 1, expansion_ratio=6) for _ in range(2)],
            
            # t=6, c=160, n=3, s=2 (then s=1)
            BottleNeckBlock(c(96), c(160), 3, stride=2, padding = 1, expansion_ratio=6),
            *[BottleNeckBlock(c(160), c(160), 3, stride=1, padding = 1, expansion_ratio=6) for _ in range(2)],
            
            # t=6, c=320, n=1, s=1
            BottleNeckBlock(c(160), c(320), 3, stride=1, padding = 1, expansion_ratio=6),
            
            # Final pointwise expansion (320 → 1280)
            ConvBlock(c(320), c(1280), 1, 1, 0),  # 7x7x1280
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(c(1280), n_classes)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, expansion_ratio: int = 6,):
        super().__init__()

        self.stride = stride
        hidden_dim = in_channels * expansion_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        # 1. Expansion (1x1 conv) - Only if expansion_ratio != 1
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace = True)]
            )
        else:
            hidden_dim = in_channels # No expansion

        # 2. Depthwise (3x3 conv)
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups = hidden_dim , bias = False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace = True)]
        )

        # 3. Projection (1x1 conv)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(out_channels),]
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)
    