import torch
from torch import nn
from ...base_models import ModelsRegistry, BaseModel

@ModelsRegistry.register("AlexNet")
class AlexNet(BaseModel):
    """
    The original adaptation of AlexNet in Caffe had a padding of 0, do to that they were using an image size of
    227x227 (eventhough at the paper they mention is 224x224).
    So if we consider using 224x224, we have to use a padding size of 2 in the first layer.

                                    (input_size + 2*padding - kernel_size)
                    output_size = ------------------------------------------- + 1
                                                    stride
    """

    def __init__(self, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            self.conv_block(3, 96, 11, 4, 2, True),
            self.conv_block(96, 256, 5, 1, 2, True),
            self.conv_block(256, 384, 3, 1, 1, False),
            self.conv_block(384, 384, 3, 1, 1, False),
            self.conv_block(384, 256, 3, 1, 1, True),
        )

        self.classifier = self.FC_block(256*6*6, 4096, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def conv_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, max_pool: bool) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        return nn.Sequential(*layers)
    
    def FC_block(self, input_size: int, neurons: int, n_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_size, neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(neurons, neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(neurons, n_classes)
        )
            