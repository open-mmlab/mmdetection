# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils import digit_version, is_tuple_of
from torch import Tensor

from mmdet.utils import MultiConfig, OptConfigType, OptMultiConfig


class SELayer(BaseModule):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Defaults to 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Defaults to (dict(type='ReLU'), dict(type='Sigmoid'))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 channels: int,
                 ratio: int = 16,
                 conv_cfg: OptConfigType = None,
                 act_cfg: MultiConfig = (dict(type='ReLU'),
                                         dict(type='Sigmoid')),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for SELayer."""
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class DyReLU(BaseModule):
    """Dynamic ReLU (DyReLU) module.

    See `Dynamic ReLU <https://arxiv.org/abs/2003.10027>`_ for details.
    Current implementation is specialized for task-aware attention in DyHead.
    HSigmoid arguments in default act_cfg follow DyHead official code.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py

    Args:
        channels (int): The input (and output) channels of DyReLU module.
        ratio (int): Squeeze ratio in Squeeze-and-Excitation-like module,
            the intermediate channel will be ``int(channels/ratio)``.
            Defaults to 4.
        conv_cfg (None or dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Defaults to (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 channels: int,
                 ratio: int = 4,
                 conv_cfg: OptConfigType = None,
                 act_cfg: MultiConfig = (dict(type='ReLU'),
                                         dict(
                                             type='HSigmoid',
                                             bias=3.0,
                                             divisor=6.0)),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.channels = channels
        self.expansion = 4  # for a1, b1, a2, b2
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels * self.expansion,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        coeffs = self.global_avgpool(x)
        coeffs = self.conv1(coeffs)
        coeffs = self.conv2(coeffs) - 0.5  # value range: [-0.5, 0.5]
        a1, b1, a2, b2 = torch.split(coeffs, self.channels, dim=1)
        a1 = a1 * 2.0 + 1.0  # [-1.0, 1.0] + 1.0
        a2 = a2 * 2.0  # [-1.0, 1.0]
        out = torch.max(x * a1 + b1, x * a2 + b2)
        return out


class ChannelAttention(BaseModule):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self, channels: int, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        if digit_version(torch.__version__) < (1, 7, 0):
            self.act = nn.Hardsigmoid()
        else:
            self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out
