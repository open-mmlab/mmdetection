import logging

import torch.nn as nn
from mmcv.runner import load_checkpoint

from ..registry import BACKBONES


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module
class MobileNetV2(nn.Module):
    """
    MobileNetV2 is taken from pytorch hub.
    https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
    """
    def __init__(self,
                 out_indices=(1, 2, 4, 6),
                 frozen_stages=-1,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],  # 0
                [6, 24, 2, 2],  # 1
                [6, 32, 3, 2],  # 2
                [6, 64, 4, 2],  # 3
                [6, 96, 3, 1],  # 4
                [6, 160, 3, 2],  # 5
                [6, 320, 1, 1],  # 6
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        assert max(out_indices) < len(inverted_residual_setting)
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.conv1 = ConvBNReLU(3, input_channel, stride=2)
        # building inverted residual blocks
        self.stages = []
        for si, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            stage = []
            for i in range(n):
                stride = s if i == 0 else 1
                stage.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            stage_name = 'stage{}'.format(si + 1)
            self.add_module(stage_name, nn.Sequential(*stage))
            self.stages.append(stage_name)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'stage{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        outs = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
