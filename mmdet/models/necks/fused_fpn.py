import torch
import torch.nn as nn

from mmdet.core import auto_fp16
from ..builder import NECKS
from .fpn import FPN


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
    )


@NECKS.register_module()
class FusedFPN(FPN):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 upscale_factors,
                 upscale_channels,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
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
                         act_cfg)

        assert num_outs == len(upscale_factors)
        self.upscale_transforms = nn.ModuleList()
        for upscale_factors in upscale_factors:
            if upscale_factors > 1:
                self.upscale_transforms.append(
                    nn.Sequential(
                        conv3x3(out_channels, upscale_channels),
                        nn.Upsample(scale_factor=upscale_factors,
                                    mode="nearest")
                    )
                )
            else:
                self.upscale_transforms.append(
                    conv3x3(out_channels, upscale_channels))

    @ auto_fp16()
    def forward(self, inputs):
        outs = super().forward(inputs)
        outs = [self.upscale_transforms[i](out) for i, out in enumerate(outs)]
        outs = torch.cat(outs, 1)
        return tuple([outs])
