import torch
import torch.nn as nn

from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint

from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import Swish


def drop_connect(x, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            Swish(),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, 
        input_width, 
        output_width,
        stride,
        exp_ratio, 
        kernel,
        se_ratio,
        dc_ratio=0.0,
        conv_cfg=None,
        norm_cfg=dict(type='BN')):
        # expansion, 3x3 dwise, BN, Swish, SE, 1x1, BN, skip_connection
        super().__init__()
        self.exp = None
        self.dc_ratio = dc_ratio
        exp_width = int(input_width * exp_ratio)
        if exp_width != input_width:
            self.exp = build_conv_layer(conv_cfg, input_width, exp_width, 1, stride=1, padding=0, bias=False)
            self.exp_bn_name, exp_bn = build_norm_layer(norm_cfg, exp_width, postfix='exp')
            self.add_module(self.exp_bn_name, exp_bn)
            self.exp_swish = Swish()
        dwise_args = {"groups": exp_width, "padding": (kernel - 1) // 2, "bias": False}
        self.dwise = build_conv_layer(conv_cfg, exp_width, exp_width, kernel, stride=stride, **dwise_args)
        self.dwise_bn_name, dwise_bn = build_norm_layer(norm_cfg, exp_width, postfix='dwise')
        self.add_module(self.dwise_bn_name, dwise_bn)
        self.dwise_swish = Swish()
        self.se = SE(exp_width, int(input_width * se_ratio))
        self.lin_proj = build_conv_layer(conv_cfg, exp_width, output_width, 1, stride=1, padding=0, bias=False) 
        self.lin_proj_bn_name, lin_proj_bn = build_norm_layer(norm_cfg, output_width, postfix='lin_proj')
        self.add_module(self.lin_proj_bn_name, lin_proj_bn)
        # Skip connection if in and out shapes are the same (MN-V2 style)
        self.has_skip = stride == 1 and input_width == output_width

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
        f_x = x
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and self.dc_ratio > 0.0:
                f_x = drop_connect(f_x, self.dc_ratio)
            f_x = x + f_x
        return f_x


class EfficientLayer(nn.Sequential):
        def __init__(
            self, 
            input_width, 
            output_width,
            stride,
            exp_ratio, 
            kernel,  
            se_ratio, 
            depth):

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
                        se_ratio=se_ratio))
            super().__init__(*layers)


@BACKBONES.register_module()
class EfficientNet(nn.Module):
    """EfficientNet model."""
    arch_settings = {
        0: ([1, 2, 2, 3, 3, 4, 1], [16, 24, 40, 80, 112, 192, 320]),
        1: ([2, 3, 3, 4, 4, 5, 2], [16, 24, 40, 80, 112, 192, 320]),
        2: ([2, 3, 3, 4, 4, 5, 2], [16, 24, 48, 88, 120, 208, 352]),
        3: ([2, 3, 3, 5, 5, 6, 2], [24, 32, 48, 96, 136, 232, 384]),
        4: ([2, 4, 4, 6, 6, 8, 2], [24, 32, 56, 112, 160, 272, 448]),
        5: ([3, 5, 5, 7, 7, 9, 3], [24, 40, 64, 128, 176, 304, 512])
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 base_channels=32,
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
                 zero_init_residual=True):
        super().__init__()
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.stage_depths, self.stage_widths = self.arch_settings[depth]

        self._make_stem_layer(3, base_channels)
        
        self.efficient_layers = []
        previous_width = base_channels
        for i, (d, w) in enumerate(zip(self.stage_depths, self.stage_widths)):
            efficient_layer = self.make_efficient_layer(
                input_width=previous_width,
                output_width=w,
                stride=strides[i],
                exp_ratio=exp_ratios[i],
                kernel=kernels[i],
                se_ratio=se_ratio,
                depth=d
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