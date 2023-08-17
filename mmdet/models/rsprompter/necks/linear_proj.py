import torch
import torch.nn as nn

# from mmpl.registry import MODELS
from mmdet.registry import MODELS


@MODELS.register_module()
class LinearProj(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=None, num_inner_layers=1):
        super(LinearProj, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_inner_layers = num_inner_layers
        if base_channels is None:
            base_channels = out_channels
        self.base_channels = base_channels

        layers = [nn.Linear(self.in_channels, self.base_channels), nn.ReLU(inplace=True)]

        for i in range(self.num_inner_layers):
            layers.append(nn.Linear(self.base_channels, self.base_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.base_channels, self.out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
