import torch
import torch.nn as nn
from mmcv.cnn import build_upsample_layer
from torch.nn import functional as fn

from mmdet.core import auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class HRNetKeypointHead(nn.Module):

    def __init__(self,
                 num_convs=8,
                 features_size=[],
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_keypoints=17,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 up_scale=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_keypoint=dict(type='MSELoss', loss_weight=1.0)):
        super(HRNetKeypointHead, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear", "carafe"'.format(
                    self.upsample_cfg['type']))
        self.num_convs = num_convs
        self.features_size = features_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor')

        self.num_keypoints = num_keypoints
        self.fp16_enabled = False
        self.loss_keypoint = build_loss(loss_keypoint)

        final_conv_kernel = 1
        self.up_scale = up_scale
        bn_momentum = 0.02
        final_inp_channels = sum(self.features_size)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=final_inp_channels,
                kernel_size=1,
                stride=1,
                padding=1 if final_conv_kernel == 3 else 0),
            nn.BatchNorm2d(final_inp_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=self.num_keypoints,
                kernel_size=final_conv_kernel,
                stride=1,
                padding=1 if final_conv_kernel == 3 else 0))

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        upsample_cfg_.update(
            in_channels=upsample_in_channels,
            out_channels=num_keypoints,
            kernel_size=self.scale_factor * 2,
            stride=self.scale_factor,
            padding=1)
        self.upsample = build_upsample_layer(upsample_cfg_)

    def init_weights(self):
        for m in [self.upsample]:
            if m is None:
                continue
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        """HR-Net keypoints head, upsampling and convolution.

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        height, width = x[0].size(2), x[0].size(3)

        x = torch.cat([
            fn.interpolate(
                x[i],
                size=(height, width),
                mode='bilinear',
                align_corners=False) for i in range(4)
        ], 1)

        x = self.conv(x)
        return x

    @force_fp32(apply_to=('keypoint_pred', 'heatmap_targets'))
    def loss(self, heatmap_pred, heatmap_targets, valids):
        loss = self.loss_keypoint(heatmap_pred, heatmap_targets)
        return {'loss_keypoints': loss}
