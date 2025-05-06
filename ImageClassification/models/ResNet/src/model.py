import torch
from torch import nn

from ...base_models import ModelsRegistry

@ModelsRegistry.register("ResNet")
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False), #[224, 224, 3] -> [112, 112, 64]
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return x

