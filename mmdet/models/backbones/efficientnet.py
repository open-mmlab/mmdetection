import mmcv
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
import warnings

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
from mmcv.cnn.bricks.drop import drop_path

from ..builder import BACKBONES
from ..utils.activations import MemoryEfficientSwish
from ..utils.inverted_residual import InvertedResidual


class MBConv(InvertedResidual):
    def __init__(self,
                 dropout=0.0,
                 **kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.dropout = dropout

    def forward(self, x):
        def _inner_forward(x):
            out = x

            if self.with_expand_conv:
                out = self.expand_conv(out)

            out = self.depthwise_conv(out)

            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)

            if self.with_res_shortcut:
                if self.dropout > 0:
                    out = drop_path(out, self.dropout, True)
                out = x + out
                return out
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class EfficientLayer(nn.Sequential):
    """EfficientLayer to build EfficientNet style backbone.

    Args:
        in_channels (int): Number of input filters.
        out_channels (int): Number of output filters.
        num_blocks (int): Number of Mobile inverted Bottleneck blocks.
        stride (int): stride of the first block.
        expand_ratio (int):
            Expansion ratios of the MBConv blocks.
        kernel_size (int):
            Kernel size of the dwise conv of the MBConv blocks.
        se_ratio (float): Ratio of the Squeeze-and-Excitation (SE) blocks.
            Default: 0.25
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks,
                 stride,
                 expand_ratio,
                 kernel_size,
                 se_ratio=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 with_cp=False,
                 dropout=0.0,
                 init_cfg=None):
        layers = []
        for d in range(num_blocks):
            block_stride = stride if d == 0 else 1
            block_width = in_channels if d == 0 else out_channels
            midchannels = int(block_width * expand_ratio)
            se_cfg = {'channels': midchannels, 'ratio': expand_ratio * se_ratio}
            with_expand_conv = False
            if midchannels != block_width:
                with_expand_conv = True
            layers.append(
                MBConv(
                    in_channels=block_width,
                    out_channels=out_channels,
                    mid_channels=midchannels,
                    stride=block_stride,
                    kernel_size=kernel_size,
                    with_cp=with_cp,
                    dropout=dropout,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    with_expand_conv=with_expand_conv,
                    init_cfg=init_cfg))
            super().__init__(*layers)


@BACKBONES.register_module()
class EfficientNet(mmcv.runner.BaseModule):
    """EfficientNet backbone.

    More details can be found in:
    `paper <https://arxiv.org/abs/1905.11946>`_ .

    Args:
        scale (int): Compund scale of EfficientNet.
            From {0, 1, 2, 3, 4, 5, 6, 7}.
        in_channels (int): Number of input image channels.
            Default: 3.
        stem_channels (int): Number of channels of the stem layer.
            Default: 32
        strides (Sequence[int]):
            Strides of the first block of each EfficientLayer.
            Default: (1, 2, 2, 2, 1, 2, 1)
        expand_ratios (Sequence[int]):
            Expansion ratios of the MBConv blocks.
            Default: (1, 6, 6, 6, 6, 6, 6)
        kernel_size (Sequence[int]):
            Kernel size for the dwise conv of the MBConv blocks.
            Default: (3, 3, 5, 3, 5, 5, 3)
        se_ratio (float): Ratio of the Squeeze-and-Excitation (SE) blocks.
            Default: 0.25
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 4, 6)
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
            Default: -1
       conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
            Default: True
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.

    Example:
        >>> from mmdet.models import EfficientNet
        >>> import torch
        >>> self = EfficientNet(scale=0)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 40, 4, 4)
        (1, 112, 2, 2)
        (1, 320, 1, 1)
    """
    arch_settings = {
        0: ([1, 2, 2, 3, 3, 4, 1], [16, 24, 40, 80, 112, 192, 320]),
        1: ([2, 3, 3, 4, 4, 5, 2], [16, 24, 40, 80, 112, 192, 320]),
        2: ([2, 3, 3, 4, 4, 5, 2], [16, 24, 48, 88, 120, 208, 352]),
        3: ([2, 3, 3, 5, 5, 6, 2], [24, 32, 48, 96, 136, 232, 384]),
        4: ([2, 4, 4, 6, 6, 8, 2], [24, 32, 56, 112, 160, 272, 448]),
        5: ([3, 5, 5, 7, 7, 9, 3], [24, 40, 64, 128, 176, 304, 512])
    }

    def __init__(self,
                 scale,
                 stem_channels=32,
                 strides=(1, 2, 2, 2, 1, 2, 1),
                 expand_ratios=(1, 6, 6, 6, 6, 6, 6),
                 kernel_size=(3, 3, 5, 3, 5, 5, 3),
                 se_ratio=4,
                 out_indices=(2, 4, 6),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 dropout=0.0,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.stage_depths, self.stage_widths = self.arch_settings[scale]
        assert scale >= 0 and scale <= 5
        assert max(out_indices) <= 6
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
        self.dropout = dropout
        self._make_stem_layer(3, stem_channels)
        self.efficient_layers = []
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        previous_width = stem_channels
        for i, (d, w) in enumerate(zip(self.stage_depths, self.stage_widths)):
            efficient_layer = self.make_efficient_layer(
                in_channels=previous_width,
                out_channels=w,
                num_blocks=d,
                stride=strides[i],
                expand_ratio=expand_ratios[i],
                kernel_size=kernel_size[i],
                se_ratio=se_ratio,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                dropout=dropout
            )
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, efficient_layer)
            self.efficient_layers.append(layer_name)
            previous_width = w

    def _make_stem_layer(self, in_channels, out_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, out_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.swish = MemoryEfficientSwish()

    def make_efficient_layer(self, **kwargs):
        return EfficientLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(EfficientNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.swish(x)
        outs = []
        for i, layer_name in enumerate(self.efficient_layers):
            efficient_layer = getattr(self, layer_name)
            x = efficient_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
