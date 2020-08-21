import torch.nn as nn

from .activations import Swish


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block with (optional) Swish.

    Args:
        w_in (int): Number of input channels.
        w_se (int): Number of squeeze channels.
        with_swish (bool): Wether to use Swish or Relu
            as intermediate activation.
    """

    def __init__(self, w_in, w_se, with_swish=True):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        activation = Swish if with_swish else nn.ReLU
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            activation(),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))
