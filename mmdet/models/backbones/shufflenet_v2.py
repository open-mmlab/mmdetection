# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, constant_init, normal_init
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.utils import channel_shuffle
from ..builder import BACKBONES


class InvertedResidual(BaseModule):
    """InvertedResidual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f'in_channels ({in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, x):

        def _inner_forward(x):
            if self.stride > 1:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


@BACKBONES.register_module()
class ShuffleNetV2(BaseModule):
    """ShuffleNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier - adjusts the number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 widen_factor=1.0,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=None):
        super(ShuffleNetV2, self).__init__(init_cfg)
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 4). But received {index}')

        if frozen_stages not in range(-1, 4):
            raise ValueError('frozen_stages must be in range(-1, 4). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
        else:
            raise ValueError('widen_factor must be in [0.5, 1.0, 1.5, 2.0]. '
                             f'But received {widen_factor}')

        self.in_channels = 24
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            layer = self._make_layer(channels[i], num_blocks)
            self.layers.append(layer)

        output_channels = channels[-1]
        self.layers.append(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=output_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def _make_layer(self, out_channels, num_blocks):
        """Stack blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): number of blocks.
        """
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(
                InvertedResidual(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(ShuffleNetV2, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv1' in name:
                    normal_init(m, mean=0, std=0.01)
                else:
                    normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m.weight, val=1, bias=0.0001)
                if isinstance(m, _BatchNorm):
                    if m.running_mean is not None:
                        nn.init.constant_(m.running_mean, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        super(ShuffleNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
