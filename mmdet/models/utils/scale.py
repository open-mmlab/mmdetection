import torch.nn as nn


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale
