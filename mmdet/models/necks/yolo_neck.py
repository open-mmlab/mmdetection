# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import load_checkpoint

from ..builder import NECKS


class DetectionNeck(nn.Module):
    """
    The DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
    1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n,
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)
    out_channels = n
    """

    def __init__(self, in_channels, out_channels):
        super(DetectionNeck, self).__init__()
        double_out_channels = out_channels * 2
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        self.conv2 = ConvModule(
            out_channels,
            double_out_channels,
            3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        self.conv3 = ConvModule(
            double_out_channels,
            out_channels,
            1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        self.conv4 = ConvModule(
            out_channels,
            double_out_channels,
            3,
            padding=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        self.conv5 = ConvModule(
            double_out_channels,
            out_channels,
            1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1))

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out


@NECKS.register_module()
class YOLOV3Neck(nn.Module):
    """
    The neck of YOLOV3.
    It can be treated as a simplified version of FPN.
    It will take the result from Darknet backbone
    and do some upsampling and concatenation.
    It will finally output the detection result.
    """

    def __init__(self):
        super(YOLOV3Neck, self).__init__()
        self.detect1 = DetectionNeck(1024, 512)
        self.conv1 = ConvModule(
            512,
            256,
            1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        self.detect2 = DetectionNeck(768, 256)
        self.conv2 = ConvModule(
            256,
            128,
            1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
        self.detect3 = DetectionNeck(384, 128)

    def forward(self, x):
        assert len(x) == 3
        x3, x2, x1 = x
        out1 = self.detect1(x1)
        tmp = self.conv1(out1)
        tmp = F.interpolate(tmp, scale_factor=2)
        tmp = torch.cat((tmp, x2), 1)
        out2 = self.detect2(tmp)
        tmp = self.conv2(out2)
        tmp = F.interpolate(tmp, scale_factor=2)
        tmp = torch.cat((tmp, x3), 1)
        out3 = self.detect3(tmp)

        return out1, out2, out3

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        else:
            raise TypeError('pretrained must be a str or None')
