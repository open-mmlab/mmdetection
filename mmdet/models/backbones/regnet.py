# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from .resnet import ResNet
from .resnext import Bottleneck


@BACKBONES.register_module()
class RegNet(ResNet):
    """RegNet backbone.

    More details can be found in `paper <https://arxiv.org/abs/2003.13678>`_ .

    Args:
        arch (dict): The parameter of RegNets.

            - w0 (int): initial width
            - wa (float): slope of width
            - wm (float): quantization parameter to quantize the width
            - depth (int): depth of the backbone
            - group_w (int): width of group
            - bot_mul (float): bottleneck ratio, i.e. expansion of bottleneck.
        strides (Sequence[int]): Strides of the first block of each stage.
        base_channels (int): Base channels after stem layer.
        in_channels (int): Number of input image channels. Default: 3.
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
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import RegNet
        >>> import torch
        >>> self = RegNet(
                arch=dict(
                    w0=88,
                    wa=26.31,
                    wm=2.25,
                    group_w=48,
                    depth=25,
                    bot_mul=1.0))
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 96, 8, 8)
        (1, 192, 4, 4)
        (1, 432, 2, 2)
        (1, 1008, 1, 1)
    """
    arch_settings = {
        'regnetx_400mf':
        dict(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22, bot_mul=1.0),
        'regnetx_800mf':
        dict(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16, bot_mul=1.0),
        'regnetx_1.6gf':
        dict(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18, bot_mul=1.0),
        'regnetx_3.2gf':
        dict(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25, bot_mul=1.0),
        'regnetx_4.0gf':
        dict(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23, bot_mul=1.0),
        'regnetx_6.4gf':
        dict(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17, bot_mul=1.0),
        'regnetx_8.0gf':
        dict(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23, bot_mul=1.0),
        'regnetx_12gf':
        dict(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, bot_mul=1.0),
    }

    def __init__(self,
                 arch,
                 in_channels=3,
                 stem_channels=32,
                 base_channels=32,
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super(ResNet, self).__init__(init_cfg)

        # Generate RegNet parameters first
        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'"arch": "{arch}" is not one of the' \
                ' arch_settings'
            arch = self.arch_settings[arch]
        elif not isinstance(arch, dict):
            raise ValueError('Expect "arch" to be either a string '
                             f'or a dict, got {type(arch)}')

        widths, num_stages = self.generate_regnet(
            arch['w0'],
            arch['wa'],
            arch['wm'],
            arch['depth'],
        )
        # Convert to per stage format
        stage_widths, stage_blocks = self.get_stages_from_blocks(widths)
        # Generate group widths and bot muls
        group_widths = [arch['group_w'] for _ in range(num_stages)]
        self.bottleneck_ratio = [arch['bot_mul'] for _ in range(num_stages)]
        # Adjust the compatibility of stage_widths and group_widths
        stage_widths, group_widths = self.adjust_width_group(
            stage_widths, self.bottleneck_ratio, group_widths)

        # Group params by stage
        self.stage_widths = stage_widths
        self.group_widths = group_widths
        self.depth = sum(stage_blocks)
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block = Bottleneck
        expansion_bak = self.block.expansion
        self.block.expansion = 1
        self.stage_blocks = stage_blocks[:num_stages]

        self._make_stem_layer(in_channels, stem_channels)

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                if self.zero_init_residual:
                    block_init_cfg = dict(
                        type='Constant', val=0, override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.inplanes = stem_channels
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            group_width = self.group_widths[i]
            width = int(round(self.stage_widths[i] * self.bottleneck_ratio[i]))
            stage_groups = width // group_width

            dcn = self.dcn if self.stage_with_dcn[i] else None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None

            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=self.stage_widths[i],
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                groups=stage_groups,
                base_width=group_width,
                base_channels=self.stage_widths[i],
                init_cfg=block_init_cfg)
            self.inplanes = self.stage_widths[i]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = stage_widths[-1]
        self.block.expansion = expansion_bak

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def generate_regnet(self,
                        initial_width,
                        width_slope,
                        width_parameter,
                        depth,
                        divisor=8):
        """Generates per block width from RegNet parameters.

        Args:
            initial_width ([int]): Initial width of the backbone
            width_slope ([float]): Slope of the quantized linear function
            width_parameter ([int]): Parameter used to quantize the width.
            depth ([int]): Depth of the backbone.
            divisor (int, optional): The divisor of channels. Defaults to 8.

        Returns:
            list, int: return a list of widths of each stage and the number \
                of stages
        """
        assert width_slope >= 0
        assert initial_width > 0
        assert width_parameter > 1
        assert initial_width % divisor == 0
        widths_cont = np.arange(depth) * width_slope + initial_width
        ks = np.round(
            np.log(widths_cont / initial_width) / np.log(width_parameter))
        widths = initial_width * np.power(width_parameter, ks)
        widths = np.round(np.divide(widths, divisor)) * divisor
        num_stages = len(np.unique(widths))
        widths, widths_cont = widths.astype(int).tolist(), widths_cont.tolist()
        return widths, num_stages

    @staticmethod
    def quantize_float(number, divisor):
        """Converts a float to closest non-zero int divisible by divisor.

        Args:
            number (int): Original number to be quantized.
            divisor (int): Divisor used to quantize the number.

        Returns:
            int: quantized number that is divisible by devisor.
        """
        return int(round(number / divisor) * divisor)

    def adjust_width_group(self, widths, bottleneck_ratio, groups):
        """Adjusts the compatibility of widths and groups.

        Args:
            widths (list[int]): Width of each stage.
            bottleneck_ratio (float): Bottleneck ratio.
            groups (int): number of groups in each stage

        Returns:
            tuple(list): The adjusted widths and groups of each stage.
        """
        bottleneck_width = [
            int(w * b) for w, b in zip(widths, bottleneck_ratio)
        ]
        groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_width)]
        bottleneck_width = [
            self.quantize_float(w_bot, g)
            for w_bot, g in zip(bottleneck_width, groups)
        ]
        widths = [
            int(w_bot / b)
            for w_bot, b in zip(bottleneck_width, bottleneck_ratio)
        ]
        return widths, groups

    def get_stages_from_blocks(self, widths):
        """Gets widths/stage_blocks of network at each stage.

        Args:
            widths (list[int]): Width in each stage.

        Returns:
            tuple(list): width and depth of each stage
        """
        width_diff = [
            width != width_prev
            for width, width_prev in zip(widths + [0], [0] + widths)
        ]
        stage_widths = [
            width for width, diff in zip(widths, width_diff[:-1]) if diff
        ]
        stage_blocks = np.diff([
            depth for depth, diff in zip(range(len(width_diff)), width_diff)
            if diff
        ]).tolist()
        return stage_widths, stage_blocks

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
