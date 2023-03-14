from typing import List

import torch
import torch.nn as nn
from mmcv.cnn.bricks import Swish
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType
from .utils import DepthWiseConvBlock, DownChannelBlock, MaxPool2dSamePadding


class BiFPNStage(nn.Module):
    """
        in_channels: List[int], input dim for P3, P4, P5
        out_channels: int, output dim for P2 - P7
        first_time: int, whether is the first bifpnstage
        conv_bn_act_pattern: bool, whether use conv_bn_act_pattern
        norm_cfg: (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer.
        epsilon: float, hyperparameter in fusion features
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 first_time: bool = False,
                 apply_bn_for_resampling: bool = True,
                 conv_bn_act_pattern: bool = False,
                 norm_cfg: OptConfigType = dict(
                     type='BN', momentum=1e-2, eps=1e-3),
                 epsilon: float = 1e-4) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_time = first_time
        self.apply_bn_for_resampling = apply_bn_for_resampling
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.norm_cfg = norm_cfg
        self.epsilon = epsilon

        if self.first_time:
            self.p5_down_channel = DownChannelBlock(
                self.in_channels[-1],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)
            self.p4_down_channel = DownChannelBlock(
                self.in_channels[-2],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)
            self.p3_down_channel = DownChannelBlock(
                self.in_channels[-3],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)
            self.p5_to_p6 = nn.Sequential(
                DownChannelBlock(
                    self.in_channels[-1],
                    self.out_channels,
                    apply_norm=self.apply_bn_for_resampling,
                    conv_bn_act_pattern=self.conv_bn_act_pattern,
                    norm_cfg=norm_cfg), MaxPool2dSamePadding(3, 2))
            self.p6_to_p7 = MaxPool2dSamePadding(3, 2)
            self.p4_level_connection = DownChannelBlock(
                self.in_channels[-2],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)
            self.p5_level_connection = DownChannelBlock(
                self.in_channels[-1],
                self.out_channels,
                apply_norm=self.apply_bn_for_resampling,
                conv_bn_act_pattern=self.conv_bn_act_pattern,
                norm_cfg=norm_cfg)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # bottom to up: feature map down_sample module
        self.p4_down_sample = MaxPool2dSamePadding(3, 2)
        self.p5_down_sample = MaxPool2dSamePadding(3, 2)
        self.p6_down_sample = MaxPool2dSamePadding(3, 2)
        self.p7_down_sample = MaxPool2dSamePadding(3, 2)

        # Fuse Conv Layers
        self.conv6_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv5_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv4_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv3_up = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv4_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv5_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv6_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
        self.conv7_down = DepthWiseConvBlock(
            out_channels,
            out_channels,
            apply_norm=self.apply_bn_for_resampling,
            conv_bn_act_pattern=self.conv_bn_act_pattern,
            norm_cfg=norm_cfg)
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

    def combine(self, x):
        if not self.conv_bn_act_pattern:
            x = self.swish(x)

        return x

    def forward(self, x):
        if self.first_time:
            p3, p4, p5 = x
            # build feature map P6
            p6_in = self.p5_to_p6(p5)
            # build feature map P7
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = x

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(
            self.combine(weight[0] * p6_in +
                         weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(
            self.combine(weight[0] * p5_in +
                         weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(
            self.combine(weight[0] * p4_in +
                         weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(
            self.combine(weight[0] * p3_in +
                         weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_level_connection(p4)
            p5_in = self.p5_level_connection(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.combine(weight[0] * p4_in + weight[1] * p4_up +
                         weight[2] * self.p4_down_sample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.combine(weight[0] * p5_in + weight[1] * p5_up +
                         weight[2] * self.p5_down_sample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.combine(weight[0] * p6_in + weight[1] * p6_up +
                         weight[2] * self.p6_down_sample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(
            self.combine(weight[0] * p7_in +
                         weight[1] * self.p7_down_sample(p6_out)))
        return p3_out, p4_out, p5_out, p6_out, p7_out


@MODELS.register_module()
class BiFPN(BaseModule):
    """
        num_stages: int, bifpn number of repeats
        in_channels: List[int], input dim for P3, P4, P5
        out_channels: int, output dim for P2 - P7
        start_level: int, Index of input features in backbone
        epsilon: float, hyperparameter in fusion features
        apply_bn_for_resampling: bool, whether use bn after resampling
        conv_bn_act_pattern: bool, whether use conv_bn_act_pattern
        norm_cfg: (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer.
        init_cfg: MultiConfig: init method
    """

    def __init__(self,
                 num_stages: int,
                 in_channels: List[int],
                 out_channels: int,
                 start_level: int = 0,
                 epsilon: float = 1e-4,
                 apply_bn_for_resampling: bool = True,
                 conv_bn_act_pattern: bool = False,
                 norm_cfg: OptConfigType = dict(
                     type='BN', momentum=1e-2, eps=1e-3),
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.start_level = start_level
        self.bifpn = nn.Sequential(*[
            BiFPNStage(
                in_channels=in_channels,
                out_channels=out_channels,
                first_time=True if _ == 0 else False,
                apply_bn_for_resampling=apply_bn_for_resampling,
                conv_bn_act_pattern=conv_bn_act_pattern,
                norm_cfg=norm_cfg,
                epsilon=epsilon) for _ in range(num_stages)
        ])

    def forward(self, x):
        x = x[self.start_level:]
        x = self.bifpn(x)

        return x
