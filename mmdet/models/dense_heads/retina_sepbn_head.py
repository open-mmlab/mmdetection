# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import bias_init_with_prob, normal_init
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig
from .anchor_head import AnchorHead


@MODELS.register_module()
class RetinaSepBNHead(AnchorHead):
    """"RetinaHead with separate BN.

    In RetinaHead, conv/norm layers are shared across different FPN levels,
    while in RetinaSepBNHead, conv layers are shared across different FPN
    levels, but BN layers are separated.
    """

    def __init__(self,
                 num_classes: int,
                 num_ins: int,
                 in_channels: int,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_ins = num_ins
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.num_ins):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for j in range(self.stacked_convs):
                chn = self.in_channels if j == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
        for i in range(self.stacked_convs):
            for j in range(1, self.num_ins):
                self.cls_convs[j][i].conv = self.cls_convs[0][i].conv
                self.reg_convs[j][i].conv = self.reg_convs[0][i].conv
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        super().init_weights()
        for m in self.cls_convs[0]:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs[0]:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward(self, feats: Tuple[Tensor]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction

                - cls_scores (list[Tensor]): Classification scores for all
                  scale levels, each is a 4D-tensor, the channels number is
                  num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all
                  scale levels, each is a 4D-tensor, the channels number is
                  num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        for i, x in enumerate(feats):
            cls_feat = feats[i]
            reg_feat = feats[i]
            for cls_conv in self.cls_convs[i]:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs[i]:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.retina_cls(cls_feat)
            bbox_pred = self.retina_reg(reg_feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds
