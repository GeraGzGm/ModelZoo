import torch
from torch import nn
from torch import Tensor, randn, zeros
from torch.nn.functional import avg_pool2d

from ...base_models import ModelsRegistry, BaseModel

@ModelsRegistry.register("LeNet1", "Classification")
class LeNet1(BaseModel):
    def __init__(self,  **kwargs):
        super().__init__()

        self.h1_w = nn.Parameter(randn(4, 24, 24, 1, 5, 5)) #4 maps, 24x24 , 1 input ch 5x5 weigths
        self.h3_w = nn.Parameter(randn(12, 8, 8, 4, 5, 5))  #12 maps, 8x8 , 4 input ch 5x5 weigths

        self.fc = nn.Linear(12 * 4 * 4, 10)


    def forward(self, x: Tensor):
        h1_h2 = self._h_hnext(x, self.h1_w)
        h3_h4 = self._h_hnext(h1_h2, self.h3_w)

        return self.linear(h3_h4)

    def _h_hnext(self, x: Tensor, weights: nn.Parameter) -> Tensor:
        maps, img_h, img_w, channels, kern_h, kern_w = weights.shape

        h = zeros( x.shape[0], maps, img_w, img_h, device = x.device )

        for m in range(maps):
            for i in range(img_h):
                for j in range(img_w):
                    h[:, m, i, j] = (x[:, :, i:i+kern_h, j:j+kern_w] * weights[m, i, j]).sum(dim = (1,2,3))
        
        return avg_pool2d(h, kernel_size=2, stride=2)

    
    def linear(self, x: Tensor) -> Tensor:
        h4_flat = x.view(x.size(0), -1)
        return self.fc(h4_flat)


if __name__ == "__main__":
    # Sanity check with random data
    model = LeNet1()
    x = torch.randn(1, 1, 28, 28)  # Fake MNIST image
    output = model(x)
    print(output.shape)  # Should be torch.Size([1, 10])