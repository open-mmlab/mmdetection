from torch.nn import Conv2d
from ..functions.masked_conv import MaskedConv2dFunction


class MaskedConv2d(Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(MaskedConv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias)

    def forward(self, input, mask):
        return MaskedConv2dFunction.apply(input, mask, self.weight, self.bias,
                                          self.padding)
