# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


class SSHModule(BaseModule):
    """This is an implementation of `SSH module` described in `SSH: Single
    Stage Headless Face Detector <https://arxiv.org/pdf/1708.03979.pdf>`_.

    Args:
        in_channels (int): Number of input channels used at each scale.
        out_channels (int): Number of output channels used at each scale.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN').
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        assert out_channels % 4 == 0

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv3X3 = ConvModule(
            self.in_channels,
            self.out_channels // 2,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.conv5X5_1 = ConvModule(
            self.in_channels,
            self.out_channels // 4,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )

        self.conv5X5_2 = ConvModule(
            self.out_channels // 4,
            self.out_channels // 4,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.conv7X7_2 = ConvModule(
            self.out_channels // 4,
            self.out_channels // 4,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )

        self.conv7X7_3 = ConvModule(
            self.out_channels // 4,
            self.out_channels // 4,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

    def forward(self, x):
        conv3X3 = self.conv3X3(x)
        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7X7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)

        return out


@MODELS.register_module()
class SSH(BaseModule):
    """`SSH Neck` used in `SSH: Single Stage Headless Face Detector.

    <https://arxiv.org/pdf/1708.03979.pdf>`_.

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (list[int]): The number of input channels per scale.
        out_channels (list[int]): The number of output channels  per scale.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN').
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_scales: int,
                 in_channels: List[int],
                 out_channels: List[int],
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 init_cfg: OptMultiConfig = dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        for idx in range(self.num_scales):
            in_c, out_c = self.in_channels[idx], self.out_channels[idx]
            self.add_module(
                f'ssh_module{idx}',
                SSHModule(
                    in_channels=in_c,
                    out_channels=out_c,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))

    def forward(self, inputs: Tuple[torch.Tensor]) -> tuple:
        assert len(inputs) == self.num_scales

        outs = []
        for idx, x in enumerate(inputs):
            ssh_module = getattr(self, f'ssh_module{idx}')
            out = ssh_module(x)
            outs.append(out)

        return tuple(outs)


@MODELS.register_module()
class RetinaSSH(BaseModule):
    """`SSH Neck` used in `RetinaFace: Single-stage Dense Face Localisation in
    the Wild <https://arxiv.org/pdf/1905.00641.pdf>`_.

    Args:
        in_channels (int): Number of input channels used at each scale.
        out_channels (int): Number of output channels used at each scale.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN').
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 init_cfg: OptMultiConfig = dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ssh_module = SSHModule(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, inputs: Tuple[torch.Tensor]) -> tuple:
        outs = []
        for x in inputs:
            out = self.ssh_module(x)
            outs.append(out)

        return tuple(outs)
