import math

import torch.nn as nn

from mmdet.ops import DeformConv, ModulatedDeformConv
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
from ..registry import BACKBONES
from ..utils import build_norm_layer


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

        self.norm1_name, norm1 = build_norm_layer(
            self.normalize, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.normalize, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.normalize, self.planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = self.dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)
        else:
            groups = self.dcn.get('groups', 1)
            deformable_groups = self.dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                width,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation)
            self.conv2 = conv_op(
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                deformable_groups=deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(
            width, self.planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   groups=1,
                   base_width=4,
                   style='pytorch',
                   with_cp=False,
                   normalize=dict(type='BN'),
                   dcn=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(normalize, planes * block.expansion)[1],
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
            with_cp=with_cp,
            normalize=normalize,
            dcn=dcn))
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
                with_cp=with_cp,
                normalize=normalize,
                dcn=dcn))

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
        normalize (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
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
            dcn = self.dcn if self.stage_with_dcn[i] else None
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
                with_cp=self.with_cp,
                normalize=self.normalize,
                dcn=dcn)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()
