import math

import torch.nn as nn

from .resnet import ResNet
from .resnet import Bottleneck as _Bottleneck
from ..registry import BACKBONES


class Bottleneck(_Bottleneck):

    def __init__(self, *args, groups=1, base_width=4, **kwargs):
        """Bottleneck block for ResNeXt.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(*args, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes * (base_width / 64)) * groups

        self.conv1 = nn.Conv2d(
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, self.planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.planes * self.expansion)


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   groups=1,
                   base_width=4,
                   style='pytorch',
                   with_cp=False):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            style=style,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                stride=1,
                dilation=dilation,
                groups=groups,
                base_width=base_width,
                style=style,
                with_cp=with_cp))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNeXt(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        super(ResNeXt, self).__init__(**kwargs)
        self.groups = groups
        self.base_width = base_width

        self.inplanes = 64
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                groups=self.groups,
                base_width=self.base_width,
                style=self.style,
                with_cp=self.with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
