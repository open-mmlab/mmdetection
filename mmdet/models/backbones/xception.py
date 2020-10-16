"""

Adepted from : https://github.com/tstandley/Xception-PyTorch
Copyright (c) 2018, Trevor Standley

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation.

Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333,
and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from mmcv.cnn import (build_conv_layer, build_norm_layer, 
                      constant_init, kaiming_init)

from ..builder import BACKBONES


class SeparableConv(nn.Module):

    def __init__(self,
                 conv_cfg,
                 in_channels,
                 out_channels,
                 kernel=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        """Sepratable Convolution Layer
        
        Simple separable convolution layer implementation.
        Used in ResNet.
        """
        super(SeparableConv, self).__init__()
        self.conv1d = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels,
            kernel,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias)
        self.pointwise = build_conv_layer(
            conv_cfg, in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):

    def __init__(self,
                 conv_cfg,
                 in_filters,
                 out_filters,
                 reps,
                 stride=1,
                 start_with_relu=True,
                 grow_first=True,
                 norm_cfg={'type': 'BN'}):
        """Basic ResNet Block
        
        Common implementation of the block unit in ResNets.
        Mostly composed of Separatable Convolution Layers. 
        """
        super(Block, self).__init__()
        self.batch_norm = False if norm_cfg is None else len(norm_cfg) != 0
        self.out_channels = out_filters
        if out_filters != in_filters or stride != 1:
            self.side = build_conv_layer(
                conv_cfg,
                in_filters,
                out_filters,
                1,
                stride=stride,
                bias=False)
            if self.batch_norm:
                _, self.bn = build_norm_layer(norm_cfg, out_filters)
        else:
            self.side = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv(
                    conv_cfg,
                    in_filters,
                    out_filters,
                    3,
                    stride=1,
                    padding=1,
                    bias=False))
            if self.batch_norm:
                rep.append(build_norm_layer(norm_cfg, out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(
                SeparableConv(
                    conv_cfg,
                    filters,
                    filters,
                    3,
                    stride=1,
                    padding=1,
                    bias=False))
            if self.batch_norm:
                rep.append(build_norm_layer(norm_cfg, filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv(
                    conv_cfg,
                    in_filters,
                    out_filters,
                    3,
                    stride=1,
                    padding=1,
                    bias=False))
            if self.batch_norm:
                rep.append(build_norm_layer(norm_cfg, out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if stride != 1:
            if conv_cfg['type'] == 'Conv3d':
                rep.append(nn.MaxPool3d(3, stride, 1))
            elif conv_cfg['type'] == 'Conv1d':
                rep.append(nn.MaxPool1d(3, stride, 1))
            else:
                rep.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.side(inp)
            if self.batch_norm:
                skip = self.bn(skip)
        else:
            skip = inp

        x += skip
        return x


@BACKBONES.register_module()
class Xception(nn.Module):
    """Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf."""

    def __init__(self,
                 depth=8,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 conv_size=[32, 64, 128, 256, 728, 1024, 1536, 2048]):
        """Xception Backend Network
        
        This is a generic implementation of the Xception network as found in:
        https://arxiv.org/pdf/1610.02357.pdf.
        This version is adapted from: https://github.com/tstandley/Xception-PyTorch
        """
        super(Xception, self).__init__()

        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.depth = depth
        self.conv_sizes = conv_size
        self.bn = False if norm_cfg is None else len(norm_cfg) != 0
        self.conv1 = build_conv_layer(
            self.conv_cfg, 3, self.conv_sizes[0], 3, 2, 0, bias=False)
        if self.bn:
            self.bn1_name, self.bn1 = build_norm_layer(self.norm_cfg,
                                                       self.conv1.out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = build_conv_layer(
            self.conv_cfg,
            self.conv1.out_channels,
            self.conv_sizes[1],
            3,
            bias=False)
        if self.bn:
            self.bn2_name, self.bn2 = build_norm_layer(self.norm_cfg,
                                                       self.conv2.out_channels)
        # do relu here
        last_ch = self.conv2.out_channels
        for blk in range(3):
            self.layer = Block(
                self.conv_cfg,
                self.conv2.out_channels,
                self.conv_sizes[2+blk],
                2,
                stride=2,
                start_with_relu=i,
                grow_first=True,
                norm_cfg=self.norm_cfg)
            last_ch = self.layer.out_channels
            self.add_module(self.layer, 'Block_{}'.format(blk))


        self.center = nn.Sequential(
            OrderedDict([('block' + i,
                          Block(
                              self.conv_cfg,
                              self.conv_sizes[4],
                              self.conv_sizes[4],
                              3,
                              stride=1,
                              start_with_relu=True,
                              grow_first=True,
                              norm_cfg=self.norm_cfg))
                         for i in range(self.depth)]))

        self.last_layer = Block(
            self.conv_cfg,
            self.conv_sizes[4],
            self.conv_sizes[5],
            2,
            stride=2,
            start_with_relu=True,
            grow_first=False,
            norm_cfg=self.norm_cfg)

        self.conv3 = SeparableConv(self.conv_cfg, self.last_layer.out_channels,
                                   self.conv_sizes[6], 3, 1, 1)
        if self.bn:
            self.bn3_name, self.bn3 = build_norm_layer(self.norm_cfg,
                                                       self.conv3.out_channels)

        # do relu here
        self.conv4 = SeparableConv(self.conv_cfg, self.conv3.out_channels,
                                   self.conv_sizes[7], 3, 1, 1)
        if self.bn:
            self.bn4_name, self.bn4 = build_norm_layer(self.norm_cfg,
                                                       self.conv4.out_channels)

        # self.fc = nn.Linear(2048, num_classes)

    def _init_weights(self, ):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                kaiming_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                constant_init(module, 1)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.center
        x = self.last_block(x)

        x = self.conv3(x)
        if self.bn:
            x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        if self.bn:
            x = self.bn4(x)
        x = self.relu(x)

        if self.conv_cfg['type'] == 'Conv3d':
            x = F.adaptive_avg_pool3d(x, (1, 1))
        elif self.conv_cfg['type'] == 'Conv1d':
            x = F.adaptive_avg_pool1d(x, (1, 1))
        else:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return tuple(x)
