# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import YOLOXHead
from mmdet.models.dense_heads.tood_head import TaskDecomposition


@HEADS.register_module()
class YOLOXTOODHead(YOLOXHead):
    """YOLOXTOOD head used in `YOLOX-PAI <https://arxiv.org/abs/2208.13040>`_.

    Args:
        tood_stacked_convs (int): Number of conv layers in TOOD head.
            Default: 3.
        la_down_rate (int): Downsample rate of layer attention.
            Default: 32.
        tood_norm_cfg (dict): Config dict for normalization layer in TOOD head.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
    """

    def __init__(self,
                 *args,
                 tood_stacked_convs=3,
                 la_down_rate=32,
                 tood_norm_cfg=dict(
                     type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.tood_stacked_convs = tood_stacked_convs
        self.la_down_rate = la_down_rate
        self.tood_norm_cfg = tood_norm_cfg

        self._build_tood_layers()

    def _build_tood_layers(self):
        self.inter_convs = nn.ModuleList()
        for _ in range(self.tood_stacked_convs):
            self.inter_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.tood_norm_cfg))

        self.multi_level_cls_decomps = nn.ModuleList()
        self.multi_level_reg_decomps = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_decomps.append(
                TaskDecomposition(self.in_channels, self.tood_stacked_convs,
                                  self.tood_stacked_convs * self.la_down_rate,
                                  self.conv_cfg, self.tood_norm_cfg))
            self.multi_level_reg_decomps.append(
                TaskDecomposition(self.in_channels, self.tood_stacked_convs,
                                  self.tood_stacked_convs * self.la_down_rate,
                                  self.conv_cfg, self.tood_norm_cfg))

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg,
                       conv_obj, cls_decomp, reg_decomp):
        """Forward feature of a single scale level."""

        inter_feats = []
        for inter_conv in self.inter_convs:
            x = inter_conv(x)
            inter_feats.append(x)
        feat = torch.cat(inter_feats, 1)

        avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        cls_x = cls_decomp(feat, avg_feat)
        reg_x = reg_decomp(feat, avg_feat)

        cls_feat = cls_convs(cls_x)
        reg_feat = reg_convs(reg_x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, objectness

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        return multi_apply(
            self.forward_single, feats, self.multi_level_cls_convs,
            self.multi_level_reg_convs, self.multi_level_conv_cls,
            self.multi_level_conv_reg, self.multi_level_conv_obj,
            self.multi_level_cls_decomps, self.multi_level_reg_decomps)
