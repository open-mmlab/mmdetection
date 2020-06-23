# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import logging

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init

from mmdet.models.utils import ConvLayer
from ..builder import BACKBONES


class ResBlock(nn.Module):
    """The basic residual block used in YoloV3.
    Each ResBlock consists of two ConvLayers and the input is added to the final output.
    In YoloV3 paper, the first convLayer has half of the number of the filters as much as the second convLayer.
    The first convLayer has filter size of 1x1 and the second one has the filter size of 3x3.
    """

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is an even number.
        half_in_channels = in_channels // 2
        self.conv1 = ConvLayer(in_channels, half_in_channels, 1)
        self.conv2 = ConvLayer(half_in_channels, in_channels, 3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        return out


def make_conv_and_res_block(in_channels, out_channels, res_repeat):
    """In Darknet 53 backbone, there is usually one Conv Layer followed by some ResBlock.
    This function will make that.
    The Conv layers always have 3x3 filters with stride=2.
    The number of the filters in Conv layer is the same as the out channels of the ResBlock"""
    model = nn.Sequential()
    model.add_module('conv', ConvLayer(in_channels, out_channels, 3, stride=2))
    for idx in range(res_repeat):
        model.add_module('res{}'.format(idx), ResBlock(out_channels))
    return model


@BACKBONES.register_module()
class DarkNet53(nn.Module):

    def __init__(self,
                 norm_eval=True,
                 reverse_output=False):
        super(DarkNet53, self).__init__()
        self.conv1 = ConvLayer(3, 32, 3)
        self.cr_block1 = make_conv_and_res_block(32, 64, 1)
        self.cr_block2 = make_conv_and_res_block(64, 128, 2)
        self.cr_block3 = make_conv_and_res_block(128, 256, 8)
        self.cr_block4 = make_conv_and_res_block(256, 512, 8)
        self.cr_block5 = make_conv_and_res_block(512, 1024, 4)

        self.norm_eval = norm_eval
        self.reverse_output=reverse_output

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.cr_block1(tmp)
        tmp = self.cr_block2(tmp)
        out3 = self.cr_block3(tmp)
        out2 = self.cr_block4(out3)
        out1 = self.cr_block5(out2)

        if not self.reverse_output:
            return out1, out2, out3
        else:
            return out3, out2, out1

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
        super(DarkNet53, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
