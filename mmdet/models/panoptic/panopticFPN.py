import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import PANOPTIC
from ..utils import ConvModule

import torch
import numpy as np
import pycocotools.mask as mask_util


@PANOPTIC.register_module
class PanopticFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_levels,
                 num_things_classes,
                 num_classes,
                 ignore_label,
                 loss_weight,
                 conv_cfg=None,
                 norm_cfg=None):
        super(PanopticFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.num_things_classes = num_things_classes
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.convP5 = nn.Sequential(
            ConvModule(self.in_channels, self.out_channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            nn.GroupNorm(self.out_channels//16, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            ConvModule(self.out_channels, self.out_channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            nn.GroupNorm(self.out_channels//16, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            ConvModule(self.out_channels, self.out_channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            nn.GroupNorm(self.out_channels//16, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.convP4 = nn.Sequential(
            ConvModule(self.in_channels, self.out_channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            nn.GroupNorm(self.out_channels//16, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            ConvModule(self.out_channels, self.out_channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            nn.GroupNorm(self.out_channels//16, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.convP3 = nn.Sequential(
            ConvModule(self.in_channels, self.out_channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            nn.GroupNorm(self.out_channels//16, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.convP2 = nn.Sequential(
            ConvModule(self.in_channels, self.out_channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            nn.GroupNorm(self.out_channels//16, self.out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_pred = nn.Sequential(
            ConvModule(self.out_channels, self.num_classes, 1, padding=0,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.num_levels

        feature_P2 = self.convP2(inputs[0])
        feature_P3 = self.convP3(inputs[1])
        feature_P4 = self.convP4(inputs[2])
        feature_P5 = self.convP5(inputs[3])
        feature_pred = feature_P2 + feature_P3 + feature_P4 + feature_P5
        feature_pred = self.conv_pred(feature_pred)

        return feature_pred

    def loss(self, segm_pred, segm_label):
        loss = dict()
        loss_segm = F.cross_entropy(segm_pred, segm_label,
                                    ignore_index=self.ignore_label)
        loss['loss_segm'] = self.loss_weight * loss_segm
        return loss

    def get_semantic_segm(self, segm_feature_pred, ori_shape,
                          img_shape_withoutpad):
        # only surport 1 batch
        segm_feature_pred = segm_feature_pred[:, :,0:img_shape_withoutpad[0],
                                              0:img_shape_withoutpad[1]]
        segm_pred_map = F.softmax(segm_feature_pred, 1)
        segm_pred_map = F.interpolate(segm_pred_map, size=ori_shape[0:2],
                                      mode="bilinear", align_corners=False)
        segm_pred_map = torch.max(segm_pred_map, 1).indices
        segm_pred_map = segm_pred_map.float()
        segm_pred_map = segm_pred_map[0]

        segm_pred_map = segm_pred_map.cpu().numpy()
        segm_pred_map_unique = np.unique(segm_pred_map).astype(np.int)
        cls_segms = [[] for _ in range(self.num_classes - 1)]

        for i in segm_pred_map_unique:
            if i <= self.num_things_classes:
                continue
            cls_im_mask = np.zeros((ori_shape[0],
                                  ori_shape[1])).astype(np.uint8)
            cls_im_mask[segm_pred_map == i] = 1
            rle = mask_util.encode(np.array(cls_im_mask[:, :, np.newaxis],
                                    order='F'))[0]
            cls_segms[i-1].append(rle)

        return cls_segms
