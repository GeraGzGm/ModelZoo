import torch
from torch import nn

from ...base_models import ModelsRegistry, BaseModel

@ModelsRegistry.register("VGG16")
class VGG16(BaseModel):
    """
    - BatchNorm was not used in the original VGG paper, but it helps to get a faster convergence. So, will be using that.

    - There's no spatial reduction in the intermediate layers (paddin = 1), until pooling layers (reducing by half).
    """

    def __init__(self, n_classes: int):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            ConvBlock(3, 64, layers = 2),    #[224, 224, 3] -> [112, 112, 64]
            ConvBlock(64, 128, layers = 2),  #[56, 56, 128]
            ConvBlock(128, 256, layers = 3), #[28, 28, 256]
            ConvBlock(256, 512, layers = 3), #[14, 14, 512]
            ConvBlock(512, 512, layers = 3), #[7, 7,512]
        )

        self.linear = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        return self.linear(x)

@ModelsRegistry.register("VGG19")
class VGG19(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            ConvBlock(3, 64, layers = 2),    #[224, 224, 3] -> [112, 112, 64]
            ConvBlock(64, 128, layers = 2),  #[56, 56, 128]
            ConvBlock(128, 256, layers = 4), #[28, 28, 256]
            ConvBlock(256, 512, layers = 4), #[14, 14, 512]
            ConvBlock(512, 512, layers = 4), #[7, 7,512]
        )

        self.linear = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        return self.linear(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int):
        super().__init__()
        
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

        following_layers = [self._make_conv_layer(out_channels, out_channels) for _ in range(layers - 1)]
        
        self.following_layers = nn.Sequential(*following_layers)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def _make_conv_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = self.following_layers(x)
        return self.max_pool(x)