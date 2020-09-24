# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import NECKS


class DetectionBlock(nn.Module):
    """Detection block in YOLO neck.

    Let out_channels = n, the DetectionBlock normally contains 5 ConvModules,
    Their sizes are 1x1xn, 3x3x2n, 1x1xn, 3x3x2n, and 1x1xn respectively.
    If the spp is on, the DetectionBlock contains 6 ConvModules and
    3 pooling layers, sizes are 1x1xn, 3x3x2n, 1x1xn,
    5x5 maxpool, 9x9 maxpool, 13x13 maxpool, 1x1xn, 3x3x2n, 1x1xn.
    The input channel is arbitrary (in_channels)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        spp_on (bool): whether to integrate a spp module. Default: False.
        spp_scales (tuple): A set of sizes for spatial pyramid pooling.
            Default: (5, 9, 13).
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 spp_on=False,
                 spp_scales=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(DetectionBlock, self).__init__()
        self.spp_on = spp_on
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)

        if self.spp_on:
            self.poolers = [
                nn.MaxPool2d(size, 1, padding=(size - 1) // 2)
                for size in spp_scales
            ]
            self.conv_spp = ConvModule(out_channels * (len(spp_scales) + 1),
                                       out_channels, 1, **cfg)

        self.conv4 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)

        if self.spp_on:
            spp_feats = [tmp] + [pooler(tmp) for pooler in self.poolers]
            tmp = torch.cat(spp_feats[::-1], 1)
            tmp = self.conv_spp(tmp)

        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out


@NECKS.register_module()
class YOLOV3Neck(nn.Module):
    """The neck of YOLOV3.

    It can be regarded as a simplified version of FPN.
    It takes the feature maps from different levels of the Darknet backbone,
    After some upsampling and concatenation, it outputs a set of
    new feature maps which are passed to the head of YOLOV3.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        spp_on (bool): Whether the spatial pyramid pooling is enabled.
            Default: False.
        spp_scales (tuple): A set of sizes for spatial pyramid pooling.
            Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 spp_on=False,
                 spp_scales=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(YOLOV3Neck, self).__init__()
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spp_on = spp_on

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # If spp is enabled, a spp block is added into the first DetectionBlock
        self.detect1 = DetectionBlock(self.in_channels[0],
                                      self.out_channels[0], spp_on, spp_scales,
                                      **cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            self.add_module(f'conv{i}', ConvModule(in_c, out_c, 1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'detect{i+1}',
                            DetectionBlock(in_c + out_c, out_c, **cfg))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        out = self.detect1(feats[-1])
        outs.append(out)

        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)

            # Cat with low-lvl feats
            tmp = F.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)

            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)

        return tuple(outs)

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass
