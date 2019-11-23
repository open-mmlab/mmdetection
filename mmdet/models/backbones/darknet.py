from collections import OrderedDict

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.registry import BACKBONES
from mmdet.models.utils import build_norm_layer


def yolo_conv2d(inplanes,
                planes,
                kernel,
                padding,
                stride,
                norm_cfg=dict(type='BN')):
    cell = OrderedDict()
    cell['conv'] = nn.Conv2d(
        inplanes,
        planes,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=False)
    if norm_cfg:
        norm_name, norm = build_norm_layer(norm_cfg, planes)
        cell[norm_name] = norm

    cell['leakyrelu'] = nn.LeakyReLU(0.1)
    cell = nn.Sequential(cell)
    return cell


class DarknetBasicBlockV3(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg=dict(type='BN')):
        super(DarknetBasicBlockV3, self).__init__()
        self.body = nn.Sequential(
            yolo_conv2d(inplanes, planes, 1, 0, 1, norm_cfg=norm_cfg),
            yolo_conv2d(planes, planes * 2, 3, 1, 1, norm_cfg=norm_cfg))

    def forward(self, x):
        residual = x
        x = self.body(x)
        return x + residual


@BACKBONES.register_module
class DarknetV3(nn.Module):

    def __init__(self,
                 layers=(1, 2, 8, 8, 4),
                 inplanes=(3, 32, 64, 128, 256, 512),
                 planes=(32, 64, 128, 256, 512, 1024),
                 num_stages=5,
                 out_indices=None,
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN'),
                 norm_eval=True):
        super(DarknetV3, self).__init__()
        assert len(layers) == len(planes) - 1, (
            "len(planes) should equal to len(layers) + 1, "
            "given {} vs {}".format(len(planes), len(layers)))
        assert 5 >= num_stages >= 1
        assert max(out_indices) < num_stages
        self.layers = layers
        self.inplanes = inplanes
        self.planes = planes
        self.num_stages = num_stages
        self.norm_cfg = norm_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self._init_layers()
        self._freeze_stages()

    def _init_layers(self):
        self.stem = yolo_conv2d(
            self.inplanes[0], self.planes[0], 3, 1, 1, norm_cfg=self.norm_cfg)

        self.darknet_layer_name = []
        for i, (nlayer, inchannel, channel) in enumerate(
                zip(self.layers[:self.num_stages],
                    self.inplanes[1:self.num_stages + 1],
                    self.planes[1:self.num_stages + 1])):
            assert channel % 2 == 0, \
                "channel {} cannot be divided by 2".format(channel)
            layer = []
            layer_name = 'layer{}'.format(i + 1)
            self.darknet_layer_name.append(layer_name)

            layer.append(
                yolo_conv2d(
                    inchannel, channel, 3, 1, 2, norm_cfg=self.norm_cfg))

            for _ in range(nlayer):
                layer.append(DarknetBasicBlockV3(channel, channel // 2))
            self.add_module(layer_name, nn.Sequential(*layer))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_state_dict(torch.load(pretrained), strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer_name in enumerate(self.darknet_layer_name):
            darknet_layer = getattr(self, layer_name)
            x = darknet_layer(x)
            if self.out_indices and i in self.out_indices:
                outs.append(x)
        return outs

    def train(self, mode=True):
        super(DarknetV3, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
