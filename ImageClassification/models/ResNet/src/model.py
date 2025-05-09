import torch
from torch import nn

from ...base_models import ModelsRegistry, BaseModel

@ModelsRegistry.register("ResNet18")
class ResNet18(BaseModel):
    def __init__(self, n_classes: int):
        super().__init__()

        self.resnet_18 = self._convolutions()

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, n_classes)
    
    def _convolutions(self) -> nn.Sequential:
        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False), #[224, 224, 3] -> [112, 112, 64]
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        conv2 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )#[56, 56, 64]

        conv3 = nn.Sequential(
            ResidualBlock(64, 128, True),
            ResidualBlock(128, 128)
        )#[28, 28, 128]

        conv4 = nn.Sequential(
            ResidualBlock(128, 256, True),
            ResidualBlock(256, 256)
        )#[14, 14, 256]

        conv5 = nn.Sequential(
            ResidualBlock(256, 512, True),
            ResidualBlock(512, 512)
        )#[7, 7, 512]
        return nn.Sequential(conv1, conv2, conv3, conv4, conv5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet_18(x)   #[7, 7, 512]
        x = self.avg_pool(x)    #[1, 1, 512]
        x = torch.flatten(x, 1) #[512]
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()

        stride = 2 if downsample else 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        #For matching dimensions (with 1x1 convolutions).
        self.projection_shorcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.projection_shorcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xo = self.conv1(x)
        xo = self.conv2(xo)

        skip = self.projection_shorcut(x)
        return nn.ReLU(inplace=True)(xo + skip)