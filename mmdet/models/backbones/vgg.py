import warnings

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES


def make_vgg_layer(in_channels,
                   out_channels,
                   num_blocks,
                   with_pool=True,
                   conv_cfg=None,
                   norm_cfg=None,
                   act_cfg=dict(type='ReLU'),
                   dilation=1,
                   ceil_mode=False):
    layers = []
    # different from mmcls, we put pooling at the beginning of a stage
    # because vgg-ssd need to use stage features before pooling
    if with_pool:
        layers.append(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    for _ in range(num_blocks):
        layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        layers.append(layer)
        in_channels = out_channels

    return layers


@BACKBONES.register_module()
class VGG(BaseModule):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
            If only one stage is specified, a single tensor (feature map) is
            returned, otherwise multiple stages are specified, a tuple of
            tensors will be returned. Default: (3, 4)
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        ceil_mode (bool): Whether to use ceil_mode of MaxPool. Default: False.
        with_last_pool (bool): Whether to keep the last pooling before
            classifier. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    # Parameters to build layers. Each element specifies the number of conv in
    # each stage. For example, VGG11 contains 11 layers with learnable
    # parameters. 11 is computed as 11 = (1 + 1 + 2 + 2 + 2) + 3,
    # where 3 indicates the last three fully-connected layers.
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }

    def __init__(self,
                 depth,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(3, 4),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 ceil_mode=False,
                 with_last_pool=False,
                 pretrained=None,
                 init_cfg=None):
        super(VGG, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
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
        else:
            raise TypeError('pretrained must be a str or None')

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert 1 <= num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages

        self.out_indices = out_indices
        if not set(out_indices).issubset(set(range(0, num_stages))):
            raise ValueError('out_indices must be a subset of range'
                             f'(0, num_stages). But received {out_indices}')

        self.frozen_stages = frozen_stages
        if frozen_stages not in range(-1, num_stages):
            raise ValueError('frozen_stages must be in range(-1, num_stages). '
                             f'But received {frozen_stages}')
        self.norm_eval = norm_eval

        self.in_channels = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks if i == 0 else num_blocks + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            out_channels = 64 * 2**i if i < 4 else 512
            vgg_layer = make_vgg_layer(
                self.in_channels,
                out_channels,
                num_blocks,
                with_pool=False if i == 0 else True,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                dilation=dilation,
                ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.in_channels = out_channels
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if with_last_pool:
            vgg_layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
            self.range_sub_modules[-1][1] += 1
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _freeze_stages(self):
        vgg_layers = getattr(self, self.module_name)
        for i in range(self.frozen_stages):
            for j in range(*self.range_sub_modules[i]):
                m = vgg_layers[j]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(VGG, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
