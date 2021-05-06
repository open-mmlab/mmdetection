import math

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet.models.builder import NECKS


@NECKS.register_module()
class CTResNetNeck(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
         in_channels (List[int]): Number of input channels per stage.
         num_filters (List[int]): Number of filters per stage.
         num_kernels (List[int]): Number of kernels per stage.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, in_channels, num_filters, num_kernels, init_cfg=None):
        super(CTResNetNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.deconv_layers = self._make_deconv_layer(num_filters, num_kernels)

    def _make_deconv_layer(self, num_filters, num_kernels):
        """use deconv layers to upsample backbone's output."""
        layers = []
        for i in range(len(num_filters)):
            feat_channels = num_filters[i]
            kenel_size = num_kernels[i]
            deform_conv = ConvModule(
                self.in_channels,
                feat_channels,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2'),
                norm_cfg=dict(type='BN'))
            layers.append(deform_conv)
            upsample_conv = nn.ConvTranspose2d(
                in_channels=feat_channels,
                out_channels=feat_channels,
                kernel_size=kenel_size,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False)
            layers.append(upsample_conv)
            layers.append(nn.BatchNorm2d(feat_channels))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = feat_channels

        return nn.Sequential(*layers)

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
        outs = self.deconv_layers(inputs[-1])
        return outs,
