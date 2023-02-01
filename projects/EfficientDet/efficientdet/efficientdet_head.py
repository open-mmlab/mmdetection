# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch.nn as nn
from mmcv.cnn.bricks import build_norm_layer
from mmengine.model import bias_init_with_prob
from torch import Tensor

from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig
from .utils import DepthWiseConvBlock, MemoryEfficientSwish


@MODELS.register_module()
class EfficientDetSepBNHead(AnchorHead):
    """EfficientDetHead with separate BN.

    num_classes (int): Number of categories excluding the background
    category. in_channels (int): Number of channels in the input feature map.
    feat_channels (int): Number of hidden channels. stacked_convs (int): Number
    of repetitions of conv norm_cfg (dict): Config dict for normalization
    layer. anchor_generator (dict): Config dict for anchor generator bbox_coder
    (dict): Config of bounding box coder. loss_cls (dict): Config of
    classification loss. loss_bbox (dict): Config of localization loss.
    train_cfg (dict): Training config of anchor head. test_cfg (dict): Testing
    config of anchor head. init_cfg (dict or list[dict], optional):
    Initialization config dict.
    """

    def __init__(self,
                 num_classes: int,
                 num_ins: int,
                 in_channels: int,
                 feat_channels: int,
                 stacked_convs: int = 3,
                 norm_cfg: OptConfigType = dict(
                     type='BN', momentum=1e-2, eps=1e-3),
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        self.num_ins = num_ins
        self.stacked_convs = stacked_convs
        self.norm_cfg = norm_cfg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            feat_channels=feat_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.reg_conv_list = nn.ModuleList()
        self.cls_conv_list = nn.ModuleList()
        for i in range(self.stacked_convs):
            channels = self.in_channels if i == 0 else self.feat_channels
            self.reg_conv_list.append(
                DepthWiseConvBlock(
                    channels, self.feat_channels, apply_norm=False))
            self.cls_conv_list.append(
                DepthWiseConvBlock(
                    channels, self.feat_channels, apply_norm=False))

        self.reg_bn_list = nn.ModuleList([
            nn.ModuleList([
                build_norm_layer(
                    self.norm_cfg, num_features=self.feat_channels)[1]
                for j in range(self.num_ins)
            ]) for i in range(self.stacked_convs)
        ])

        self.cls_bn_list = nn.ModuleList([
            nn.ModuleList([
                build_norm_layer(
                    self.norm_cfg, num_features=self.feat_channels)[1]
                for j in range(self.num_ins)
            ]) for i in range(self.stacked_convs)
        ])

        self.cls_header = DepthWiseConvBlock(
            self.in_channels,
            self.num_base_priors * self.cls_out_channels,
            apply_norm=False)
        self.reg_header = DepthWiseConvBlock(
            self.in_channels, self.num_base_priors * 4, apply_norm=False)
        self.swish = MemoryEfficientSwish()

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.reg_conv_list:
            nn.init.constant_(m.pointwise_conv.conv.bias, 0.0)
        for m in self.cls_conv_list:
            nn.init.constant_(m.pointwise_conv.conv.bias, 0.0)
        bias_cls = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_header.pointwise_conv.conv.bias, bias_cls)
        nn.init.constant_(self.reg_header.pointwise_conv.conv.bias, 0.0)

    def forward_single_bbox(self, feat: Tensor, level_id: int,
                            i: int) -> Tensor:
        conv_op = self.reg_conv_list[i]
        bn = self.reg_bn_list[i][level_id]

        feat = conv_op(feat)
        feat = bn(feat)
        feat = self.swish(feat)

        return feat

    def forward_single_cls(self, feat: Tensor, level_id: int,
                           i: int) -> Tensor:
        conv_op = self.cls_conv_list[i]
        bn = self.cls_bn_list[i][level_id]

        feat = conv_op(feat)
        feat = bn(feat)
        feat = self.swish(feat)

        return feat

    def forward(self, feats: Tuple[Tensor]) -> tuple:
        cls_scores = []
        bbox_preds = []
        for level_id in range(self.num_ins):
            feat = feats[level_id]
            for i in range(self.stacked_convs):
                feat = self.forward_single_bbox(feat, level_id, i)
            bbox_pred = self.reg_header(feat)
            bbox_preds.append(bbox_pred)
        for level_id in range(self.num_ins):
            feat = feats[level_id]
            for i in range(self.stacked_convs):
                feat = self.forward_single_cls(feat, level_id, i)
            cls_score = self.cls_header(feat)
            cls_scores.append(cls_score)

        return cls_scores, bbox_preds
