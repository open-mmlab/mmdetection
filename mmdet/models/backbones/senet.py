import math

import torch.nn as nn
import torch.utils.checkpoint as cp
from collections import OrderedDict

from mmdet.ops import DeformConv, ModulatedDeformConv
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
from .resnext import Bottleneck as ResNeXtBottleneck


class SEModule(nn.Module):
    def __init__(self, channels, reduction, conv_cfg=None):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = build_conv_layer(
            conv_cfg,
            channels,
            channels // reduction,
            kernel_size=1,
            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = build_conv_layer(
            conv_cfg,
            channels // reduction,
            channels,
            kernel_size=1,
            padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return identity * x


class _SEBottleneck(nn.Module):
    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out_ = self.conv1(x)
            out_ = self.norm1(out_)
            out_ = self.relu(out_)

            if not self.with_dcn:
                out_ = self.conv2(out_)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out_)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                out_ = self.conv2(out_, offset, mask)
            else:
                offset = self.conv2_offset(out_)
                out_ = self.conv2(out_, offset)
            out_ = self.norm2(out_)
            out_ = self.relu(out_)

            out_ = self.conv3(out_)
            out_ = self.norm3(out_)

            if self.downsample is not None:
                identity = self.downsample(x)

            out_ = self.se_module(out_) + identity

            return out_

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        return out


class SEBottleneck(_SEBottleneck, _Bottleneck):
    """
       Bottleneck for SENet154.
    """
    def __init__(self, *args, groups=64, reduction=16, **kwargs):
        super(SEBottleneck, self).__init__(*args, **kwargs)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.planes * 2, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, self.planes * 4, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            self.planes * 2,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = self.dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                self.conv_cfg,
                self.planes * 2,
                self.planes * 4,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            groups = self.dcn.get('groups', 1)
            deformable_groups = self.dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                self.planes * 2,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation)
            self.conv2 = conv_op(
                self.planes * 2,
                self.planes * 4,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                deformable_groups=deformable_groups,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            self.planes * 4,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(self.planes * self.expansion, reduction=reduction)


class SEResNetBottleneck(_SEBottleneck, _Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    def __init__(self, *args, groups=1, reduction=16, **kwargs):
        super(SEResNetBottleneck, self).__init__(*args, **kwargs)

        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = self.dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                self.conv_cfg,
                self.planes,
                self.planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            deformable_groups = self.dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                self.planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation)
            self.conv2 = conv_op(
                self.planes,
                self.planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                deformable_groups=deformable_groups,
                bias=False)
        self.se_module = SEModule(self.planes * self.expansion, reduction=reduction)


class SEResNeXtBottleneck(_SEBottleneck, ResNeXtBottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    def __init__(self, *args, groups=32, reduction=16, base_width=4, **kwargs):
        super(SEResNeXtBottleneck, self).__init__(*args, groups=groups, base_width=base_width, **kwargs)
        self.se_module = SEModule(self.planes * self.expansion, reduction=reduction)


def make_se_layer(block,
                  inplanes,
                  planes,
                  blocks,
                  stride=1,
                  dilation=1,
                  groups=1,
                  reduction=16,
                  with_cp=False,
                  conv_cfg=None,
                  norm_cfg=dict(type='BN'),
                  dcn=None,
                  downsample_kernel_size=1,
                  downsample_padding=0,
                  base_width=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=downsample_kernel_size,
                padding=downsample_padding,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = list()
    if not base_width:
        layers.append(
            block(
                inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                groups=groups,
                reduction=reduction,
                base_width=base_width,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
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
                    reduction=reduction,
                    base_width=base_width,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn))
    else:
        assert block == SEResNeXtBottleneck, "argument 'base_width' must for SEResNeXtBottleneck"
        layers.append(
            block(
                inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                groups=groups,
                reduction=reduction,
                base_width=base_width,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
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
                    reduction=reduction,
                    base_width=base_width,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class SeNet(ResNet):
    """SeNet backbone.

    Args:
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
    """
    arch_settings = {154: (SEBottleneck, [3, 8, 36, 3])}

    def __init__(self,
                 inplanes=128,
                 input_3x3=True,
                 groups=64,
                 reduction=16,
                 downsample_kernel_sizes=(1, 3, 3, 3),
                 downsample_paddings=(0, 1, 1, 1),
                 base_width=None,
                 **kwargs):

        self.input_3x3 = input_3x3
        super(SeNet, self).__init__(**kwargs)
        self.inplanes = inplanes
        del self.res_layers

        self.groups = groups
        self.reduction = reduction
        self.se_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            planes = 64 * 2**i
            se_layer = make_se_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                groups=groups,
                reduction=reduction,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                downsample_kernel_size=downsample_kernel_sizes[i],
                downsample_padding=downsample_paddings[i],
                base_width=base_width)

            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i+1)
            self.add_module(layer_name, se_layer)
            self.se_layers.append(layer_name)

        self._freeze_stages()

    def _make_stem_layer(self):
        # just for compatibility parent method
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix='')
        if self.input_3x3:
            _, norm2 = build_norm_layer(self.norm_cfg, 64, postfix='')
            _, norm3 = build_norm_layer(self.norm_cfg, self.inplanes, postfix='')
            stem_layer = [
                ('conv1', build_conv_layer(self.conv_cfg, 3,
                                           64, kernel_size=3, stride=2, padding=1, bias=False)),
                ('norm1', norm1),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', build_conv_layer(self.conv_cfg, 64,
                                           64, kernel_size=3, stride=1, padding=1, bias=False)),
                ('norm2', norm2),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', build_conv_layer(self.conv_cfg, 64,
                                           self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)),
                ('norm3', norm3),
                ('relu3', nn.ReLU(inplace=True))
            ]
        else:
            stem_layer = [
                ('conv1', build_conv_layer(self.conv_cfg, 3,
                                           64, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm1', norm1),
                ('relu1', nn.ReLU(inplace=True))
            ]
        stem_layer.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(stem_layer))
        self.add_module('layer0', self.layer0)

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.layer0(x)
        outs = []
        for i, layer_name in enumerate(self.se_layers):
            se_layer = getattr(self, layer_name)
            x = se_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module
class SeResNet(SeNet):
    arch_settings = {
        50: (SEResNetBottleneck, [3, 4, 6, 3]),
        101: (SEResNetBottleneck, [3, 4, 23, 3]),
        152: (SEResNetBottleneck, [3, 8, 36, 3])
    }

    def __init__(self,
                 inplanes=64,
                 input_3x3=False,
                 groups=1,
                 reduction=16,
                 downsample_kernel_sizes=(1, 1, 1, 1),
                 downsample_paddings=(0, 0, 0, 0),
                 **kwargs):
        super(SeResNet, self).__init__(inplanes,
                                       input_3x3,
                                       groups,
                                       reduction,
                                       downsample_kernel_sizes,
                                       downsample_paddings,
                                       **kwargs)


@BACKBONES.register_module
class SeResNeXt(SeNet):
    arch_settings = {
        50: (SEResNeXtBottleneck, [3, 4, 6, 3]),
        101: (SEResNeXtBottleneck, [3, 4, 23, 3]),
        152: (SEResNeXtBottleneck, [3, 8, 36, 3])
    }

    def __init__(self,
                 inplanes=64,
                 input_3x3=False,
                 groups=32,
                 reduction=16,
                 base_width=4,
                 downsample_kernel_sizes=(1, 1, 1, 1),
                 downsample_paddings=(0, 0, 0, 0),
                 **kwargs):
        super(SeResNeXt, self).__init__(inplanes=inplanes,
                                        input_3x3=input_3x3,
                                        groups=groups,
                                        reduction=reduction,
                                        base_width=base_width,
                                        downsample_kernel_sizes=downsample_kernel_sizes,
                                        downsample_paddings=downsample_paddings,
                                        **kwargs)