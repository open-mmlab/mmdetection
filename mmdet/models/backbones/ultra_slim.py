import logging

import torch.nn as nn
from mmcv.runner import load_checkpoint
from torch.nn import Conv2d, ModuleList, ReLU, Sequential

from ..registry import BACKBONES


class Base_net(nn.Module):

    def __init__(self, num_classes=2):
        super(Base_net, self).__init__()
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, self.base_channel, 2),  # 160*120
            conv_dw(self.base_channel, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 2, 2),  # 80*60
            conv_dw(self.base_channel * 2, self.base_channel * 2, 1),
            conv_dw(self.base_channel * 2, self.base_channel * 4, 2),  # 40*30
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 4, 1),
            conv_dw(self.base_channel * 4, self.base_channel * 8, 2),  # 20*15
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 8, 1),
            conv_dw(self.base_channel * 8, self.base_channel * 16, 2),  # 10*8
            conv_dw(self.base_channel * 16, self.base_channel * 16, 1))

    def forward(self, x):
        x = self.model(x)
        return x


source_layer_indexes = [8, 11, 13]


@BACKBONES.register_module
class UltraSlim(nn.Module):

    def __init__(self, input_size):
        super(UltraSlim, self).__init__()
        self.base_net = Base_net()
        self.source_layer_indexes = source_layer_indexes

        self.extras = ModuleList([
            Sequential(
                Conv2d(
                    in_channels=self.base_net.base_channel * 16,
                    out_channels=self.base_net.base_channel * 4,
                    kernel_size=1), ReLU(),
                SeperableConv2d(
                    in_channels=self.base_net.base_channel * 4,
                    out_channels=self.base_net.base_channel * 16,
                    kernel_size=3,
                    stride=2,
                    padding=1), ReLU())
        ])

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        start_layer_index = 0
        end_layer_index = 0
        for end_layer_index in self.source_layer_indexes:
            for layer in self.base_net.model[
                    start_layer_index:end_layer_index]:
                x = layer(x)
            y = x
            start_layer_index = end_layer_index
            outs.append(y)
        for layer in self.base_net.model[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)


def xavier_init(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def SeperableConv2d(in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding),
        ReLU(),
        Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )
