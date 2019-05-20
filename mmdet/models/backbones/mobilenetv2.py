import torch
import torch.nn as nn
from mmcv.cnn import (constant_init, kaiming_init, normal_init)
from ..registry import BACKBONES


def conv_bn(inp, oup, stride, groups=1, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )


def conv_1x1_bn(inp, oup, groups=1, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, activation=nn.ReLU):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        inp = int(inp)
        oup = int(oup)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = [
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module
class SSDMobilenetV2(nn.Module):
    def __init__(self, input_size, width_mult=1.0, activation_type='relu'):
        super(SSDMobilenetV2, self).__init__()
        self.input_size = input_size

        self.width_mult = width_mult
        block = InvertedResidual
        input_channel = 32
        last_channel = 480
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 1],
            [4, 480, 1, 1],
        ]
        assert activation_type in ['relu', 'relu6']
        if activation_type in 'relu':
            self.activation_class = nn.ReLU
        else:
            self.activation_class = nn.ReLU6

        # building first layer
        input_channel = int(input_channel * self.width_mult)
        if self.width_mult > 1.0:
            self.last_channel = int(last_channel * self.width_mult)
        else:
            self.last_channel = last_channel
        self.bn_first = nn.BatchNorm2d(3)
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = c * self.width_mult
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel,
                                               s, t, self.activation_class))
                else:
                    self.features.append(block(input_channel, output_channel,
                                               1, t, self.activation_class))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            if self.width_mult != 1.0:
                patched_dict = {}
                for k, v in state_dict.items():
                    if 'backbone.' in k:
                        k = k[len('backbone.'):]
                    if 'conv' in k:  # process convs in inverted residuals
                        if len(v.shape) == 1:
                            v = v[:int(v.shape[0]*self.width_mult)]
                        elif len(v.shape) == 4 and v.shape[1] == 1:
                            assert v.shape[2] == v.shape[3] and v.shape[2] == 3
                            v = v[:int(v.shape[0]*self.width_mult), ]
                        elif len(v.shape) == 4 and v.shape[2] == 1:
                            assert v.shape[2] == v.shape[3] and v.shape[2] == 1
                            v = v[:int(v.shape[0]*self.width_mult),
                                  :int(v.shape[1]*self.width_mult), ]
                    elif 'features.0.' in k:  # process the first conv
                        if len(v.shape):
                            v = v[:int(v.shape[0]*self.width_mult), ]

                    patched_dict[k] = v

                for k, v in state_dict.items():
                    if 'features.17.conv' in k:
                        del patched_dict[k]
                self.load_state_dict(patched_dict, strict=False)
            else:
                self.load_state_dict(state_dict, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        x = self.bn_first(x)
        for i, block in enumerate(self.features):
            x = block(x)
        outs.append(x)
        return tuple(outs)
