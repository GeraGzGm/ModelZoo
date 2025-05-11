import torch
from torch import nn

from ...base_models import ModelsRegistry, BaseModel

@ModelsRegistry.register("InceptionV1")
class InceptionV1(BaseModel):
    AUX_LOSS_DISCOUNT = 0.3

    def __init__(self, n_classes: int, **kwargs):
        super().__init__()

        self.cv1 = self.block()
        self.inception_3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)       #[28,28,256]
        self.inception_3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)     #[28,28,480]

        self.inception_4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)      #[14,14,512]
        self.inception_4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)     #[14,14,512]
        self.inception_4b_classifier = AuxiliaryClassifier(512, n_classes)
        self.inception_4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)     #[14,14,512]
        self.inception_4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)     #[14,14,528]
        self.inception_4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)   #[14,14,832]
        self.inception_4e_classifier = AuxiliaryClassifier(528, n_classes)

        self.inception_5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)   #[7,7,832]
        self.inception_5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)   #[7,7,1024]

        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.classifier = nn.Sequential(
            #nn.AvgPool2d(kernel_size = 7, stride = 1, padding = 0), #[1,1,1024]
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Dropout(p = 0.4),
            nn.Linear(1024, n_classes)
        )

    def block(self) -> nn. Sequential:
        return nn.Sequential(
            BaseConv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),

            BaseConv2d(64, 64, 1, 1, 0),

            BaseConv2d(64, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        x = self.cv1(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool(x)

        x = self.inception_4a(x)

        if self.training:
            inception_4b_classifier = self.inception_4b_classifier(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)

        if self.training:
            inception_4e_classifier = self.inception_4e_classifier(x)
        x = self.inception_4e(x)
        x = self.maxpool(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.classifier(x)

        if not self.training:
            return x
        
        return {
            "main": x,
            "aux1": inception_4b_classifier,
            "aux2": inception_4e_classifier
        }

    def compute_loss(self, outputs: dict, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """
        During training, the auxiliary losses are added to the main loss with a discount weight (0.3).
        """
        loss_main = criterion(outputs["main"], targets)
        loss_aux1 = criterion(outputs["aux1"], targets)
        loss_aux2 = criterion(outputs["aux2"], targets)
        return loss_main + self.AUX_LOSS_DISCOUNT * (loss_aux1 + loss_aux2)

    def get_main_output(self, outputs: dict | torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, dict):
            return outputs["main"]
        return outputs

class BaseConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

class InceptionModule(nn.Module):
    """Inception module with dimension reductions."""
    def __init__(self, in_channels: int, c1x1: int, c3x3_red: int, c3x3: int, c5x5_red: int, c5x5: int, pool_proj: int):
        super().__init__()

        self.branch1 = nn.Conv2d(in_channels, c1x1, kernel_size = 1, stride = 1, padding = 0)

        self.branch2 = nn.Sequential(
            BaseConv2d(in_channels, c3x3_red, kernel_size = 1, stride = 1, padding = 0),
            BaseConv2d(c3x3_red, c3x3, kernel_size = 3, stride = 1, padding = 1)
        )

        self.branch3 = nn.Sequential(
            BaseConv2d(in_channels, c5x5_red, kernel_size = 1, stride = 1, padding = 0),
            BaseConv2d(c5x5_red, c5x5, kernel_size = 5, stride = 1, padding = 2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            BaseConv2d(in_channels, pool_proj, kernel_size = 1, stride = 1, padding = 0)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
    
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size = 5, stride = 3, padding = 0)             #[4,4,512] | [4,4,528]
        self.conv = BaseConv2d(in_channels, 128, kernel_size = 1, stride = 1, padding = 0) #[4,4,128] -> 2048

        self.linear = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.7),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.conv(x)
        x = torch.flatten(x, start_dim = 1)
        return self.linear(x)