import torch
import torch.nn as nn
from mmdet.core import auto_fp16

from .fpn import FPN
from ..registry import NECKS
from ..utils import ConvModule


class RSSH(nn.Module):
    def __init__(self, in_channels, conv_cfg, norm_cfg, activation):
        super(RSSH, self).__init__()
        self.in_channels = in_channels
        self.activation = activation

        self.conv1 = ConvModule(
            in_channels,
            in_channels // 2,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=self.activation,
            inplace=False)

        self.conv2 = ConvModule(
            in_channels // 2,
            in_channels // 4,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=self.activation,
            inplace=False)

        self.conv3 = ConvModule(
            in_channels // 4,
            in_channels // 4,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=self.activation,
            inplace=False)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        return torch.cat((x1, x2, x3), axis=1)


@NECKS.register_module
class RSSH_FPN(FPN):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super().__init__(in_channels,
                         out_channels,
                         num_outs,
                         start_level,
                         end_level,
                         add_extra_convs,
                         extra_convs_on_inputs,
                         relu_before_extra_convs,
                         no_norm_on_lateral,
                         conv_cfg,
                         norm_cfg,
                         activation)

        self.context_modules = \
            nn.ModuleList(
                [RSSH(out_channels, conv_cfg, norm_cfg, activation) for _ in self.fpn_convs])

    @auto_fp16()
    def forward(self, inputs):
        outs = super().forward(inputs)
        outs = [self.context_modules[i](out) for i, out in enumerate(outs)]
        return tuple(outs)
