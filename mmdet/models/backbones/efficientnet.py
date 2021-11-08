import torch.nn as nn
import torch
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from ..utils.activations import Swish
from ..utils.se_block import SE


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    with torch.no_grad():
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor)  # binarize
    output = x / keep_prob * random_tensor
    return output


class MBConv(nn.Module):
    """Mobile inverted Bottleneck block with Squeeze-and-Excitation (SE).

    Args:
        input_width (int): Number of input filters.
        output_width (int): Number of output filters.
        stride (int): stride of the first block.
        exp_ratio (int): Expansion ratio..
        kernel (int): Kernel size of the dwise conv.
        se_ratio (float): Ratio of the Squeeze-and-Excitation (SE).
            Default: 0.25
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
    """

    def __init__(self,
                 input_width,
                 output_width,
                 stride,
                 exp_ratio,
                 kernel,
                 se_ratio=0.25,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 with_cp=False,
                 dropout=0.0):
        super().__init__()
        self.exp = None
        self.with_cp = with_cp
        self.dropout = dropout
        exp_width = int(input_width * exp_ratio)
        if exp_width != input_width:
            self.exp = build_conv_layer(
                conv_cfg,
                input_width,
                exp_width,
                1,
                stride=1,
                padding=0,
                bias=False)
            self.exp_bn_name, exp_bn = build_norm_layer(
                norm_cfg, exp_width, postfix='exp')
            self.add_module(self.exp_bn_name, exp_bn)
            self.exp_swish = Swish()
        dwise_args = {
            'groups': exp_width,
            'padding': (kernel - 1) // 2,
            'bias': False
        }
        self.dwise = build_conv_layer(
            conv_cfg,
            exp_width,
            exp_width,
            kernel,
            stride=stride,
            **dwise_args)
        self.dwise_bn_name, dwise_bn = build_norm_layer(
            norm_cfg, exp_width, postfix='dwise')
        self.add_module(self.dwise_bn_name, dwise_bn)
        self.dwise_swish = Swish()
        self.se = SE(exp_width, int(input_width * se_ratio))
        self.lin_proj = build_conv_layer(
            conv_cfg,
            exp_width,
            output_width,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.lin_proj_bn_name, lin_proj_bn = build_norm_layer(
            norm_cfg, output_width, postfix='lin_proj')
        self.add_module(self.lin_proj_bn_name, lin_proj_bn)
        # Skip connection if in and out shapes are the same (MN-V2 style)
        self.has_skip = (stride == 1 and input_width == output_width)

    @property
    def dwise_bn(self):
        return getattr(self, self.dwise_bn_name)

    @property
    def exp_bn(self):
        return getattr(self, self.exp_bn_name)

    @property
    def lin_proj_bn(self):
        return getattr(self, self.lin_proj_bn_name)

    def forward(self, x):
        def _inner_forward(x):
            f_x = x
            if self.exp:
                f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
            f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
            f_x = self.se(f_x)
            f_x = self.lin_proj_bn(self.lin_proj(f_x))
            if self.has_skip:
                if self.dropout > 0:
                    f_x = drop_path(f_x, self.dropout, True)
                f_x = x + f_x

            return f_x

        if self.with_cp and x.requires_grad:
            f_x = cp.checkpoint(_inner_forward, x)
        else:
            f_x = _inner_forward(x)

        return f_x


class EfficientLayer(nn.Sequential):
    """EfficientLayer to build EfficientNet style backbone.

    Args:
        input_width (int): Number of input filters.
        output_width (int): Number of output filters.
        depth (int): Number of Mobile inverted Bottleneck blocks.
        stride (int): stride of the first block.
        exp_ratio (int):
            Expansion ratios of the MBConv blocks.
        kernel (int):
            Kernel size of the dwise conv of the MBConv blocks.
        se_ratio (float): Ratio of the Squeeze-and-Excitation (SE) blocks.
            Default: 0.25
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
    """

    def __init__(self,
                 input_width,
                 output_width,
                 depth,
                 stride,
                 exp_ratio,
                 kernel,
                 se_ratio=0.25,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 with_cp=False,
                 dropout=0.0
                 ):
        layers = []
        for d in range(depth):
            block_stride = stride if d == 0 else 1
            block_width = input_width if d == 0 else output_width
            layers.append(
                MBConv(
                    input_width=block_width,
                    output_width=output_width,
                    stride=block_stride,
                    exp_ratio=exp_ratio,
                    kernel=kernel,
                    se_ratio=se_ratio,
                    with_cp=with_cp,
                    dropout=dropout))
            super().__init__(*layers)


@BACKBONES.register_module()
class EfficientNet(nn.Module):
    """EfficientNet backbone.

    More details can be found in:
    `paper <https://arxiv.org/abs/1905.11946>`_ .

    Args:
        scale (int): Compund scale of EfficientNet.
            From {0, 1, 2, 3, 4, 5, 6, 7}.
        in_channels (int): Number of input image channels.
            Default: 3.
        base_channels (int): Number of channels of the stem layer.
            Default: 32
        strides (Sequence[int]):
            Strides of the first block of each EfficientLayer.
            Default: (1, 2, 2, 2, 1, 2, 1)
        exp_ratios (Sequence[int]):
            Expansion ratios of the MBConv blocks.
            Default: (1, 6, 6, 6, 6, 6, 6)
        kernels (Sequence[int]):
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
                 in_channels=3,
                 base_channels=40,
                 strides=(1, 2, 2, 2, 1, 2, 1),
                 exp_ratios=(1, 6, 6, 6, 6, 6, 6),
                 kernels=(3, 3, 5, 3, 5, 5, 3),
                 se_ratio=0.25,
                 out_indices=(2, 4, 6),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 with_cp=False,
                 dropout=0.0
                 ):
        super().__init__()
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.stage_depths, self.stage_widths = self.arch_settings[scale]
        self.dropout = dropout
        # self.dropout = nn.Dropout(dropout)
        self._make_stem_layer(3, base_channels)
        self.efficient_layers = []
        previous_width = base_channels
        for i, (d, w) in enumerate(zip(self.stage_depths, self.stage_widths)):
            efficient_layer = self.make_efficient_layer(
                input_width=previous_width,
                output_width=w,
                depth=d,
                stride=strides[i],
                exp_ratio=exp_ratios[i],
                kernel=kernels[i],
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
        self.swish = Swish()

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

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

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
