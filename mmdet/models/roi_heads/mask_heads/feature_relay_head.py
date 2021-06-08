import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import HEADS


@HEADS.register_module()
class FeatureRelayHead(BaseModule):
    """Feature Relay Head used in `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        in_channels (int, optional): number of input channels. Default: 256.
        conv_out_channels (int, optional): number of output channels before
            classification layer. Default: 256.
        roi_feat_size (int, optional): roi feat size at box head. Default: 7.
        scale_factor (int, optional): scale factor to match roi feat size
            at mask head. Default: 2.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=1024,
                 out_conv_channels=256,
                 roi_feat_size=7,
                 scale_factor=2,
                 init_cfg=dict(type='Kaiming', layer='Linear')):
        super(FeatureRelayHead, self).__init__(init_cfg)
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

    @auto_fp16()
    def forward(self, x):
        """Forward function."""
        N, in_C = x.shape
        if N > 0:
            out_C = self.out_conv_channels
            out_HW = self.roi_feat_size
            x = self.fc(x)
            x = x.reshape(N, out_C, out_HW, out_HW)
            x = self.upsample(x)
            return x
        return None
