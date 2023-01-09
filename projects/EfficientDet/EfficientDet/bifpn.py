import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Swish
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType


class DepthWiseConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            conv_cfg: OptConfigType = dict(type='Conv2dAdaptivePadding'),
            norm_cfg: OptConfigType = dict(type='BN', momentum=1e-2, eps=1e-3),
            fuse_act_cfg: OptConfigType = dict(type='Swish'),
    ) -> None:
        super(DepthWiseConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            groups=in_channels,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=None,
            act_cfg=None)

        self.base_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=fuse_act_cfg)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.base_conv(x)
        return x


class MaxPool2dSamePadding(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) -
                   1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) -
                   1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


@MODELS.register_module()
class BiFPN(BaseModule):
    '''
        in_channels: List[int], input dim for P3, P4, P5
        out_channels: int, output dim for P2 - P7
        num_outs: int, BiFPN need feature maps num
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to Conv2dAdaptivePadding.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    '''

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        start_level: int = 0,
        epsilon: float = 1e-4,
        conv_cfg: OptConfigType = dict(type='Conv2dAdaptivePadding'),
        norm_cfg: OptConfigType = dict(type='BN', momentum=1e-2, eps=1e-3),
        act_cfg: OptConfigType = None,
        fuse_act_cfg: OptConfigType = dict(type='Swish'),
        upsample_cfg: OptConfigType = dict(scale_factor=2, mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.fp16_enabled = False
        self.epsilon = epsilon
        self.upsample_cfg = upsample_cfg.copy()

        # build P6 feature map
        self.p5_to_p6 = nn.Sequential(
            ConvModule(
                in_channels[-1],
                out_channels,
                1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg), MaxPool2dSamePadding(3, 2))
        # build P7 feature map
        self.p6_to_p7 = nn.Sequential(MaxPool2dSamePadding(3, 2))

        # down_channels for P3, P4, P5
        self.p3_down_channel = ConvModule(
            in_channels[0],
            out_channels,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.p4_down_channel = ConvModule(
            in_channels[1],
            out_channels,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.p5_down_channel = ConvModule(
            in_channels[2],
            out_channels,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # same level skip connection
        self.p4_level_connection = ConvModule(
            in_channels[1],
            out_channels,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.p5_level_connection = ConvModule(
            in_channels[2],
            out_channels,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # bottom to up: feature map down_sample module
        self.p4_down_sample = MaxPool2dSamePadding(3, 2)
        self.p5_down_sample = MaxPool2dSamePadding(3, 2)
        self.p6_down_sample = MaxPool2dSamePadding(3, 2)
        self.p7_down_sample = MaxPool2dSamePadding(3, 2)

        # Fuse Conv Layers
        self.conv6_up = DepthWiseConvBlock(
            out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            fuse_act_cfg=fuse_act_cfg)
        self.conv5_up = DepthWiseConvBlock(
            out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            fuse_act_cfg=fuse_act_cfg)
        self.conv4_up = DepthWiseConvBlock(
            out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            fuse_act_cfg=fuse_act_cfg)
        self.conv3_up = DepthWiseConvBlock(
            out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            fuse_act_cfg=fuse_act_cfg)
        self.conv4_down = DepthWiseConvBlock(
            out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            fuse_act_cfg=fuse_act_cfg)
        self.conv5_down = DepthWiseConvBlock(
            out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            fuse_act_cfg=fuse_act_cfg)
        self.conv6_down = DepthWiseConvBlock(
            out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            fuse_act_cfg=fuse_act_cfg)
        self.conv7_down = DepthWiseConvBlock(
            out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            fuse_act_cfg=fuse_act_cfg)

        # weights
        self.p6_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.swish = Swish()

    def forward(self, x):
        p3, p4, p5 = x

        # build feature map P6
        p6_in = self.p5_to_p6(p5)
        # build feature map P7
        p7_in = self.p6_to_p7(p6_in)

        p3_in = self.p3_down_channel(p3)
        p4_in = self.p4_down_channel(p4)
        p5_in = self.p5_down_channel(p5)

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(
            self.swish(weight[0] * p6_in +
                       weight[1] * F.interpolate(p7_in, **self.upsample_cfg)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(
            self.swish(weight[0] * p5_in +
                       weight[1] * F.interpolate(p6_up, **self.upsample_cfg)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(
            self.swish(weight[0] * p4_in +
                       weight[1] * F.interpolate(p5_up, **self.upsample_cfg)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(
            self.swish(weight[0] * p3_in +
                       weight[1] * F.interpolate(p4_up, **self.upsample_cfg)))

        p4_in = self.p4_level_connection(p4)
        p5_in = self.p5_level_connection(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up +
                       weight[2] * self.p4_down_sample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up +
                       weight[2] * self.p5_down_sample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up +
                       weight[2] * self.p6_down_sample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(
            self.swish(weight[0] * p7_in +
                       weight[1] * self.p7_down_sample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out
