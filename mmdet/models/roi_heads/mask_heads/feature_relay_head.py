# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig


@MODELS.register_module()
class FeatureRelayHead(BaseModule):
    """Feature Relay Head used in `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        in_channels (int): number of input channels. Defaults to 256.
        conv_out_channels (int): number of output channels before
            classification layer. Defaults to 256.
        roi_feat_size (int): roi feat size at box head. Default: 7.
        scale_factor (int): scale factor to match roi feat size
            at mask head. Defaults to 2.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict. Defaults to
            dict(type='Kaiming', layer='Linear').
    """

    def __init__(
        self,
        in_channels: int = 1024,
        out_conv_channels: int = 256,
        roi_feat_size: int = 7,
        scale_factor: int = 2,
        init_cfg: MultiConfig = dict(type='Kaiming', layer='Linear')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(roi_feat_size, int)

        self.in_channels = in_channels
        self.out_conv_channels = out_conv_channels
        self.roi_feat_size = roi_feat_size
        self.out_channels = (roi_feat_size**2) * out_conv_channels
        self.scale_factor = scale_factor
        self.fp16_enabled = False

        self.fc = nn.Linear(self.in_channels, self.out_channels)
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x: Tensor) -> Optional[Tensor]:
        """Forward function.

        Args:
            x (Tensor): Input feature.

        Returns:
            Optional[Tensor]: Output feature. When the first dim of input is
            0, None is returned.
        """
        N, _ = x.shape
        if N > 0:
            out_C = self.out_conv_channels
            out_HW = self.roi_feat_size
            x = self.fc(x)
            x = x.reshape(N, out_C, out_HW, out_HW)
            x = self.upsample(x)
            return x
        return None
