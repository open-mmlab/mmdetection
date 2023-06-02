# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class SSDNeck(BaseModule):
    """SSD 主干的额外卷积层以生成多尺度特征图.

    Args:
        in_channels (Sequence[int]): 多层级特征的输入通道数(512, 1024).
        out_channels (Sequence[int]): 多层级特征的输出通道数(512, 1024, 512, 256, 256, 256).
        level_strides (Sequence[int]): 每层级上3x3卷积的stride.
        level_paddings (Sequence[int]): 每层级上3x3卷积的padding.
        l2_norm_scale (float|None): L2 归一化层初始scale.
            如果为 None,则不对第一个输入特征使用 L2 归一化.
        last_kernel_size (int): 最后一个卷积层的卷积核大小.
        use_depthwise (bool): 是否使用深度可分离卷积.
        conv_cfg (dict): 卷积层的配置字典. Default: None.
        norm_cfg (dict): 构造和配置norm层的字典.
        act_cfg (dict): 激活层的配置字典.
        init_cfg (dict or list[dict], optional): 初始化配置字典.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 level_strides,
                 level_paddings,
                 l2_norm_scale=20.,
                 last_kernel_size=3,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=[
                     dict(
                         type='Xavier', distribution='uniform',
                         layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d'),
                 ]):
        super(SSDNeck, self).__init__(init_cfg)
        assert len(out_channels) > len(in_channels)
        assert len(out_channels) - len(in_channels) == len(level_strides)
        assert len(level_strides) == len(level_paddings)
        assert in_channels == out_channels[:len(in_channels)]

        if l2_norm_scale:
            self.l2_norm = L2Norm(in_channels[0], l2_norm_scale)
            self.init_cfg += [
                dict(
                    type='Constant',
                    val=self.l2_norm.scale,
                    override=dict(name='l2_norm'))
            ]
        # SSD的neck层是由两个stride=2,padding=1的conv和两个stride=1,padding=0的conv组成的
        # 以此来达到"下采样"4次的作用
        self.extra_layers = nn.ModuleList()
        extra_layer_channels = out_channels[len(in_channels):]
        second_conv = DepthwiseSeparableConvModule if \
            use_depthwise else ConvModule

        for i, (out_channel, stride, padding) in enumerate(
                zip(extra_layer_channels, level_strides, level_paddings)):
            kernel_size = last_kernel_size if i == len(extra_layer_channels) - 1 else 3
            per_lvl_convs = nn.Sequential(
                ConvModule(
                    out_channels[len(in_channels) - 1 + i],
                    out_channel // 2,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                second_conv(
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
        # 因为tuple为不可变量,所以这里需要更改为list类型
        outs = [feat for feat in inputs]
        if hasattr(self, 'l2_norm'):
            outs[0] = self.l2_norm(outs[0])

        feat = outs[-1]
        for layer in self.extra_layers:
            feat = layer(feat)
            outs.append(feat)
        return tuple(outs)


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): 要归一化的维数
            scale (float, optional): 默认为 20.
            eps (float, optional): 用于避免除以零.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # 归一化层在训练状态下,会强制转换为FP32进行操作,最后再转为输入类型返回
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)
