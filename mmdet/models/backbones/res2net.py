import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class Bottle2neck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 scale=4,
                 base_width=26,
                 stype='normal'):
        """Bottle2neck block for Res2Net.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottle2neck, self).__init__(
            inplanes,
            planes,
            stride=stride,
            dilation=stride,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            plugins=plugins)

        width = int(math.floor(planes * (base_width / 64.0)))

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, width * scale, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            width * scale,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage' and self.conv2_stride != 1:
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []

        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            for i in range(self.nums):
                convs.append(
                    build_conv_layer(
                        conv_cfg,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=dilation,
                        dilation=dilation,
                        bias=False))
                bns.append(build_norm_layer(norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            for i in range(self.nums):
                convs.append(
                    build_conv_layer(
                        dcn,
                        width,
                        width,
                        kernel_size=3,
                        stride=self.conv2_stride,
                        padding=dilation,
                        dilation=dilation,
                        bias=False))
                bns.append(build_norm_layer(norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)

        self.conv3 = build_conv_layer(
            conv_cfg,
            width * scale,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.stype = stype
        self.scale = scale
        self.width = width
        delattr(self, 'conv2')
        delattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            spx = torch.split(out, self.width, 1)
            for i in range(self.nums):
                if i == 0 or self.stype == 'stage':
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp)
                sp = self.relu(self.bns[i](sp))
                if i == 0:
                    out = sp
                else:
                    out = torch.cat((out, sp), 1)
            if (self.scale != 1 and self.stype == 'normal') \
                    or self.conv2_stride == 1:
                out = torch.cat((out, spx[self.nums]), 1)
            elif self.scale != 1 and self.stype == 'stage':
                out = torch.cat((out, self.pool(spx[self.nums])), 1)

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


class ResLayer(nn.Sequential):
    """ResLayer to build Res2Net style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        scale (int): Scales used in Res2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 26
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 scale=4,
                 base_width=26,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False),
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                scale=scale,
                base_width=base_width,
                stype='stage',
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    scale=scale,
                    base_width=base_width,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class Res2Net(ResNet):
    """Res2Net backbone.

    Args:
        scale (int): Scales used in Res2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 26
        depth (int): Depth of res2net, from {50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Res2net stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import Res2Net
        >>> import torch
        >>> self = Res2Net(depth=50, scale=4, base_width=26)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 8, 8)
        (1, 512, 4, 4)
        (1, 1024, 2, 2)
        (1, 2048, 1, 1)
    """

    arch_settings = {
        50: (Bottle2neck, (3, 4, 6, 3)),
        101: (Bottle2neck, (3, 4, 23, 3)),
        152: (Bottle2neck, (3, 8, 36, 3))
    }

    def __init__(self, scale=4, base_width=26, **kwargs):
        self.scale = scale
        self.base_width = base_width
        kwargs['style'] = 'pytorch'
        kwargs['deep_stem'] = True
        kwargs['avg_down'] = True
        super(Res2Net, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(scale=self.scale, base_width=self.base_width, **kwargs)
