import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import HEADS
from ..builder import build_loss
from ..utils import ConvModule

import torch
import numpy as np
import pycocotools.mask as mask_util


@HEADS.register_module
class SemanticSegmHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 num_things_classes,
                 num_classes,
                 ignore_label,
                 loss_semantic_segm,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticSegmHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.num_things_classes = num_things_classes
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_semantic_segm = build_loss(loss_semantic_segm)

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = ConvModule(
                    self.in_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    one_conv = ConvModule(
                        self.in_channels,
                        self.out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue

                one_conv = ConvModule(
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            ConvModule(
                self.out_channels,
                self.num_classes,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                activation=None),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level + 1)

        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            feature_add_all_level += self.convs_all_levels[i](inputs[i])
        feature_pred = self.conv_pred(feature_add_all_level)

        return feature_pred

    def loss(self, segm_pred, segm_label):
        loss = dict()
        loss['loss_segm'] = self.loss_semantic_segm(
            segm_pred, segm_label, ignore_label=self.ignore_label)
        return loss

    def get_semantic_segm(self, segm_feature_pred, ori_shape,
                          img_shape_withoutpad):
        # only surport 1 batch
        segm_feature_pred = segm_feature_pred[:, :, 0:img_shape_withoutpad[0],
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
            if i == 0:
                continue
            cls_im_mask = np.zeros(
                (ori_shape[0], ori_shape[1])).astype(np.uint8)
            cls_im_mask[segm_pred_map == i] = 1
            rle = mask_util.encode(
                np.array(cls_im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[i-1].append(rle)

        return cls_segms
