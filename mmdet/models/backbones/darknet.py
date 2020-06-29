# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import logging

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init, ConvModule

from ..builder import BACKBONES


class ResBlock(nn.Module):
    """The basic residual block used in YoloV3.
    Each ResBlock consists of two ConvModules and the input is added to the final output.
    Each ConvModule is composed of Conv, BN, and LeakyReLU
    In YoloV3 paper, the first convLayer has half of the number of the filters as much as the second convLayer.
    The first convLayer has filter size of 1x1 and the second one has the filter size of 3x3.
    """

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is an even number.
        half_in_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels,
                                half_in_channels,
                                1,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        self.conv2 = ConvModule(half_in_channels, 
                                in_channels, 
                                3,
                                padding=1,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        # self.conv1 = ConvLayer(in_channels, half_in_channels, 1)
        # self.conv2 = ConvLayer(half_in_channels, in_channels, 3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        return out


def make_conv_and_res_block(in_channels, out_channels, res_repeat):
    """In Darknet backbone, there is usually one Conv Layer followed by some ResBlock.
    This function will make that.
    The Conv layers always have 3x3 filters with stride=2.
    The number of the filters in Conv layer is the same as the out channels of the ResBlock"""
    model = nn.Sequential()
    model.add_module('conv', ConvModule(in_channels,
                                        out_channels,
                                        3,
                                        stride=2,
                                        padding=1,
                                        norm_cfg=dict(type='BN', requires_grad=True),
                                        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)))
    for idx in range(res_repeat):
        model.add_module('res{}'.format(idx), ResBlock(out_channels))
    return model


@BACKBONES.register_module()
class Darknet(nn.Module):
    """Darknet backbone.

    Args:
        depth (int): Depth of Darknet. Currently only support 53.
        out_indices (Sequence[int]): Output from which stages. 
            Note: By default, the sequence of the layers will be returned
            in a **reversed** manner. i.e., from bottom to up.
            See the example bellow.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        reverse_output (bool): If True, the sequence of the output layers
            will be from bottom to up. Default: True. (To cope with YoloNeck)

    Example:
        >>> from mmdet.models import Darknet
        >>> import torch
        >>> self = Darknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 1024, 13, 13)
        (1, 512, 26, 26)
        (1, 256, 52, 52)
    """

    def __init__(self,
                 depth=53,
                 out_indices=(3, 4, 5),
                 norm_eval=True,
                 reverse_output=True):
        super(Darknet, self).__init__()
        self.depth = depth
        self.out_indices = out_indices
        if self.depth == 53:
            self.layers = [1, 2, 8, 8, 4]
            self.channels = [[32, 64], [64, 128], [128, 256], [256, 512], [512, 1024]]
        else:
            raise KeyError(f'invalid depth {depth} for darknet')

        self.conv1 = ConvModule(3, 32, 3,
                                padding=1,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                act_cfg=dict(type='LeakyReLU', negative_slope=0.1))

        self.cr_blocks = ['conv1']
        for i, n_layers in enumerate(self.layers):
            layer_name = f'cr_block{i + 1}'
            in_c, out_c = self.channels[i]
            self.add_module(layer_name, make_conv_and_res_block(in_c, out_c, n_layers))
            self.cr_blocks.append(layer_name)

        self.norm_eval = norm_eval
        self.reverse_output=reverse_output

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        if self.reverse_output:
            return tuple(outs[::-1])
        else:
            return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super(Darknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
