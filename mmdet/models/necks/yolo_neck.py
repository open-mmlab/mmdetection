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

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(DetectionNeck, self).__init__()
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)
        self.conv4 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

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

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(YOLOV3Neck, self).__init__()
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # shortcut
        cfg = dict(norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.detect1 = DetectionNeck(in_channels[0], out_channels[0], **cfg)
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            self.add_module(f'conv{i}', ConvModule(in_c, out_c, 1, **cfg))
            self.add_module(f'detect{i+1}',
                            DetectionNeck(in_c + out_c, out_c, **cfg))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        outs = []

        out = self.detect1(feats[-1])
        outs.append(out)
        for i, x in enumerate(feats[1::-1]):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)
            tmp = F.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)

            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)

        return tuple(outs)

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
