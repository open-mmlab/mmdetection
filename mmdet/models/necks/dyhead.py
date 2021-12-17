import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, build_activation_layer, build_norm_layer,
                      constant_init, normal_init)
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmcv.runner import BaseModule

from ..builder import NECKS

# Reference:
# https://github.com/microsoft/DynamicHead
# https://github.com/jshilong/SEPC

# We follow official code for hard-sigmoid function.
# paper's hard-sigmoid corresponds to HSigmoid(bias=1.0, divisor=2.0)
# code's hard-sigmoid corresponds to HSigmoid(bias=3.0, divisor=6.0)
# https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py


class DYReLU(BaseModule):
    """DYReLU module for Task-aware Attention in DyHead.

    Dynamic ReLU https://arxiv.org/abs/2003.10027

    Args:
        in_channels (int): The input channels of the DYReLU module.
        out_channels (int): The output channels of the DYReLU module.
        ratio (int): Squeeze ratio in Squeeze-and-Excitation-like module,
            the intermediate channel will be ``int(channels/ratio)``.
            Default: 4.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ratio=4,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                          dict(type='HSigmoid', bias=3.0, divisor=6.0)),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = 4  # for a1, b1, a2, b2
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=int(in_channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(in_channels / ratio),
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        """Forward function."""
        # forward Squeeze-and-Excitation-like module
        coeffs = self.global_avgpool(x)
        coeffs = self.conv1(coeffs)
        coeffs = self.conv2(coeffs)  # [0.0, 1.0] if default HSigmoid
        # split DYReLU coefficients and normalize them
        a1, b1, a2, b2 = torch.split(coeffs, self.out_channels, dim=1)
        a1 = (a1 - 0.5) * 2.0 + 1.0  # [-1.0, 1.0] + 1.0
        a2 = (a2 - 0.5) * 2.0  # [-1.0, 1.0]
        b1 = b1 - 0.5  # [-0.5, 0.5]
        b2 = b2 - 0.5  # [-0.5, 0.5]
        out = torch.max(x * a1 + b1, x * a2 + b2)
        return out


class MDCN3x3Norm(nn.Module):
    """ModulatedDeformConv2d with normalization layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        x = self.norm(x)
        return x


class DyHeadBlock(nn.Module):
    """DyHead Block."""

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        self.spatial_conv_high = MDCN3x3Norm(in_channels, out_channels)
        self.spatial_conv_med = MDCN3x3Norm(in_channels, out_channels)
        self.spatial_conv_low = MDCN3x3Norm(
            in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 1, 1),
            nn.ReLU(inplace=True))
        self.scale_attn_sigmoid = build_activation_layer(act_cfg)
        self.task_attn_module = DYReLU(in_channels, out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level, feature in enumerate(x):
            # calculate offset and mask of DCNv2 from median-level feature
            offset_and_mask = self.spatial_conv_offset(feature)
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            # spatial-aware attention
            mlvl_feats = [self.spatial_conv_med(feature, offset, mask)]
            if level > 0:
                mlvl_feats.append(
                    self.spatial_conv_low(x[level - 1], offset, mask))
            if level < len(x) - 1:
                mlvl_feats.append(
                    F.interpolate(
                        self.spatial_conv_high(x[level + 1], offset, mask),
                        size=feature.shape[-2:],
                        mode='bilinear',
                        align_corners=True))

            # scale-aware attention and task-aware attention
            res_feat = torch.stack(mlvl_feats)
            attn_feats = [self.scale_attn_conv(feat) for feat in mlvl_feats]
            scale_attn = self.scale_attn_sigmoid(torch.stack(attn_feats))
            mean_feat = torch.mean(res_feat * scale_attn, dim=0, keepdim=False)
            outs.append(self.task_attn_module(mean_feat))

        return outs


@NECKS.register_module()
class DyHead(BaseModule):
    """DyHead neck consisting of multiple DyHead Blocks.

    https://arxiv.org/abs/2106.08322
    """

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 stacked_convs=6,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stacked_convs = stacked_convs

        dyhead_blocks = []
        for i in range(stacked_convs):
            in_channels = self.in_channels if i == 0 else self.out_channels
            dyhead_blocks.append(DyHeadBlock(in_channels, self.out_channels))
        self.dyhead_blocks = nn.Sequential(*dyhead_blocks)

    def forward(self, inputs):
        """Forward function."""
        outs = self.dyhead_blocks(inputs)
        return tuple(outs)
