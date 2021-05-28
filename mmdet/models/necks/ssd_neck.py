import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class SSDNeck(BaseModule):
    """Extra layers of SSD backbone to generate multi-scale feature maps.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): Number of output channels per scale.
        level_strides (List[int]): Stride of 3x3 conv per level.
        level_paddings (List[int]): Padding size of 3x3 conv per level.
        l2_norm_scale (float): L2 normalization layer init scale.
            If None, not use L2 normalization on the first input feature.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels=[512, 1024],
                 out_channels=[512, 1024, 512, 256, 256, 256],
                 level_strides=[2, 2, 1, 1],
                 level_paddings=[1, 1, 0, 0],
                 l2_norm_scale=20.,
                 last_kernel_size=3,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d'),
                     dict(type='Normal', std=0.01, layer='Linear')
                 ]):
        super(SSDNeck, self).__init__(init_cfg)
        if l2_norm_scale:
            self.l2_norm = L2Norm(in_channels[0], l2_norm_scale)

        self.extra_layers = nn.ModuleList()
        extra_layer_channels = out_channels[len(in_channels):]
        Conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        for i, (out_channel, stride, padding) in enumerate(
                zip(extra_layer_channels, level_strides, level_paddings)):
            kernel_size = last_kernel_size \
                if i == len(extra_layer_channels) - 1 else 3
            per_lvl_convs = nn.Sequential(
                ConvModule(
                    out_channels[len(in_channels) - 1 + i],
                    out_channel // 2,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                Conv(
                    out_channel // 2,
                    out_channel,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.extra_layers.append(per_lvl_convs)

    def forward(self, inputs):
        """Forward function."""
        outs = [feat for feat in inputs]
        if hasattr(self, 'l2_norm'):
            outs[0] = self.l2_norm(outs[0])

        feat = inputs[-1]
        for layer in self.extra_layers:
            feat = layer(feat)
            outs.append(feat)
        return tuple(outs)


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)
