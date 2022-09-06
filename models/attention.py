import math
from torch import nn, Tensor
from torch.nn import functional as F
import torch

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ModifiedCA(nn.Module):

    def __init__(self, input_dim):
        super(ModifiedCA, self).__init__()
        self.c_mlp = MLP(input_dim * 2, input_dim, input_dim, 3)
        self.ins_mlp = MLP(input_dim, input_dim, input_dim, 3)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor, y: Tensor):
        # feature descriptor on the global spatial information
        DEVICE = x.device
        z = torch.cat([x, y], dim=-1)
        z = self.c_mlp(z).sigmoid()

        x_res = self.ins_mlp(x)

        y_ca_aug = y + x_res + y * z

        return y_ca_aug