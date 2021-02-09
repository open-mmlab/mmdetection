import math

import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2dPack

from mmdet.models.builder import NECKS

BN_MOMENTUM = 0.1


@NECKS.register_module()
class CT_ResNeck(nn.Module):
    """The head used in CenterNet for object classification and box regression.

    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, in_channels, num_filters, num_kernels):
        super(CT_ResNeck, self).__init__()
        self.in_channels = in_channels
        self.deconv_layers = self._make_deconv_layer(num_filters, num_kernels,
                                                     BN_MOMENTUM)

    def _make_deconv_layer(self, num_filters, num_kernels, BN_MOMENTUM):
        assert len(num_filters) == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(len(num_filters)):
            feat_channels = num_filters[i]
            kenel_size = num_kernels[i]
            deform = ModulatedDeformConv2dPack(
                self.in_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                deformable_groups=1)
            up = nn.ConvTranspose2d(
                in_channels=feat_channels,
                out_channels=feat_channels,
                kernel_size=kenel_size,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False)
            layers.append(deform)
            layers.append(nn.BatchNorm2d(feat_channels, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(feat_channels, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = feat_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = x[-1]
        x = self.deconv_layers(x)
        return x

    def init_weights(self, pretrained=True):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ModulatedDeformConv2dPack):
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


# if __name__ == '__main__':
#     model = CT_ResNeck(3, 3, [256, 128, 64], [4, 4, 4]).cuda()
#     input = torch.rand(2, 3, 40, 40).cuda()

#     print(model)
#     output = model(input)
#     print(output.shape)
#     model.init_weights()
