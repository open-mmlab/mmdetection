import math

import numpy as np
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2dPack

from mmdet.models.builder import NECKS

BN_MOMENTUM = 0.1


class DeformConv(nn.Module):

    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))
        self.conv = ModulatedDeformConv2dPack(
            chi,
            cho,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(
                o,
                o,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=o,
                bias=False)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):

    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


@NECKS.register_module()
class CTDLANetNeck(nn.Module):

    def __init__(self,
                 in_channels=[16, 32, 64, 128, 256, 512],
                 start_level=2,
                 end_level=5):
        super(CTDLANetNeck, self).__init__()
        self.start_level = start_level
        self.end_level = end_level
        scales = [2**i for i in range(len(in_channels[self.start_level:]))]
        self.dla_up = DLAUp(self.start_level, in_channels[self.start_level:],
                            scales)
        self.ida_up = IDAUp(
            in_channels[self.start_level],
            in_channels[self.start_level:self.end_level],
            [2**i for i in range(self.end_level - self.start_level)])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x = self.dla_up(inputs)
        y = []
        for i in range(self.end_level - self.start_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        return y[-1],
