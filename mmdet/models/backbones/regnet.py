import numpy as np
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class Bottleneck(_Bottleneck):
    expansion = 1

    def __init__(self, inplanes, planes, group_width=8, **kwargs):
        """Bottleneck block for RegNetX.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)
        width = int(round(planes * self.expansion))
        groups = width // group_width
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                self.conv_cfg,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                self.dcn,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg, width, self.planes, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)


@BACKBONES.register_module()
class RegNet(ResNet):
    """RegNet backbone.

    Args:
        arch_parameter (dict):
        strides (Sequence[int]): Strides of the first block of each stage.
        in_channels (int): Number of input image channels. Normally 3.
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

    Example:
        >>> from mmdet.models import ResNeXt
        >>> import torch
        >>> self = ResNeXt(depth=50)
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

    def __init__(self,
                 depth,
                 arch_parameter,
                 strides=(2, 2, 2, 2),
                 base_channels=32,
                 **kwargs):
        widths, num_stages = self.generate_regnet(
            arch_parameter['w0'],
            arch_parameter['wa'],
            arch_parameter['wm'],
            depth,
        )
        # Convert to per stage format
        stage_widths, stage_depths = self.get_stages_from_blocks(
            widths, widths)
        # Generate group widths and bot muls
        group_widths = [arch_parameter['group_w'] for _ in range(num_stages)]
        self.bottleneck_ratio = [
            arch_parameter['bot_mul'] for _ in range(num_stages)
        ]
        # Adjust the compatibility of stage_widths and group_widths
        stage_widths, group_widths = self.adjust_width_group(
            stage_widths, self.bottleneck_ratio, group_widths)
        # Use the same stride for each stage

        # Group params by stage
        self.stage_widths = stage_widths
        self.group_widths = group_widths
        self.arch_settings = {
            sum(stage_depths): (Bottleneck, tuple(stage_depths))
        }

        super(RegNet, self).__init__(
            depth=depth,
            strides=strides,
            base_channels=base_channels,
            **kwargs)

        self.inplanes = base_channels
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            self.block.expansion = int(self.bottleneck_ratio[i])
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
                group_width=self.group_widths[i])
            self.inplanes = self.stage_widths[i]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = stage_widths[-1]

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
        """Generates per block width from RegNet parameters."""

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
        """Converts a float to closest non-zero int divisible by q."""
        return int(round(number / divisor) * divisor)

    def adjust_width_group(self, widths, bottleneck_ratio, groups):
        """Adjusts the compatibility of widths and groups."""
        ws_bot = [int(w * b) for w, b in zip(widths, bottleneck_ratio)]
        groups = [min(g, w_bot) for g, w_bot in zip(groups, ws_bot)]
        ws_bot = [
            self.quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, groups)
        ]
        widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratio)]
        return widths, groups

    def get_stages_from_blocks(self, widths, rs):
        """Gets widths/stage_depths of network at each stage"""
        ts_temp = zip(widths + [0], [0] + widths, rs + [0], [0] + rs)
        ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
        stage_widths = [w for w, t in zip(widths, ts[:-1]) if t]
        stage_depths = np.diff([d for d, t in zip(range(len(ts)), ts)
                                if t]).tolist()
        return stage_widths, stage_depths

    def forward(self, x):
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
