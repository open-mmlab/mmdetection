# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, Sequential
from torch import Tensor
from torch import nn as nn

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Defaults to 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Defaults to None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Defaults to dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Defaults to True
    """

    def __init__(self,
                 block: BaseModule,
                 inplanes: int,
                 planes: int,
                 num_blocks: int,
                 stride: int = 1,
                 avg_down: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 downsample_first: bool = True,
                 **kwargs) -> None:
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super().__init__(*layers)


class SimplifiedBasicBlock(BaseModule):
    """Simplified version of original basic residual block. This is used in
    `SCNet <https://arxiv.org/abs/2012.10150>`_.

    - Norm layer is now optional
    - Last ReLU in forward function is removed
    """
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[Sequential] = None,
                 style: ConfigType = 'pytorch',
                 with_cp: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 dcn: OptConfigType = None,
                 plugins: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert not with_cp, 'Not implemented yet.'
        self.with_norm = norm_cfg is not None
        with_bias = True if norm_cfg is None else False
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=with_bias)
        if self.with_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, planes, postfix=1)
            self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=with_bias)
        if self.with_norm:
            self.norm2_name, norm2 = build_norm_layer(
                norm_cfg, planes, postfix=2)
            self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self) -> Optional[BaseModule]:
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name) if self.with_norm else None

    @property
    def norm2(self) -> Optional[BaseModule]:
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name) if self.with_norm else None

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for SimplifiedBasicBlock."""

        identity = x

        out = self.conv1(x)
        if self.with_norm:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
