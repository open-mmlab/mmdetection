import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNetV1d


class rSoftMax(nn.Module):

    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d."""

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 radix=2,
                 reduction_factor=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None):
        super(SplAtConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.with_dcn = dcn is not None
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if self.with_dcn and not fallback_on_stride:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            conv_cfg = dcn
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            channels * radix,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups * radix,
            bias=bias)
        self.norm0_name, norm0 = build_norm_layer(
            norm_cfg, channels * radix, postfix=0)
        self.add_module(self.norm0_name, norm0)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = build_conv_layer(
            conv_cfg, channels, inter_channels, 1, groups=self.cardinality)
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, inter_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.fc2 = build_conv_layer(
            conv_cfg,
            inter_channels,
            channels * radix,
            1,
            groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    @property
    def norm0(self):
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class Bottleneck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 radix=2,
                 reduction_factor=4,
                 avd=True,
                 **kwargs):
        """Bottleneck block for ResNeSt.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups

        self.avd = avd and self.conv2_stride > 1

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.with_modulated_dcn = False
        self.conv2 = SplAtConv2d(
            width,
            width,
            kernel_size=3,
            stride=1 if self.avd else self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            radix=radix,
            reduction_factor=reduction_factor,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            dcn=self.dcn)
        delattr(self, self.norm2_name)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, self.conv2_stride, padding=1)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)

            if self.avd:
                out = self.avd_layer(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNeSt(ResNetV1d):
    """ResNeSt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
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
        norm_cfg (dict): dictionary to construct and config norm layer.
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

    def __init__(self,
                 groups=1,
                 base_width=4,
                 radix=2,
                 reduction_factor=4,
                 avd=True,
                 **kwargs):
        self.groups = groups
        self.base_width = base_width
        self.radix = radix
        self.reduction_factor = reduction_factor
        self.avd = avd
        super(ResNeSt, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            radix=self.radix,
            reduction_factor=self.reduction_factor,
            avd=self.avd,
            **kwargs)
