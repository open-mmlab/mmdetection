# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads import CenterNetUpdateHead
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2distance
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)

INF = 1000000000
RangeType = Sequence[Tuple[int, int]]


@MODELS.register_module()
class CenterNetRPNHead(CenterNetUpdateHead):
    """CenterNetUpdateHead is an improved version of CenterNet in CenterNet2.
    Paper link `<https://arxiv.org/abs/2103.07461>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channel in the input feature map.
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        hm_min_radius (int): Heatmap target minimum radius of cls branch.
            Defaults to 4.
        hm_min_overlap (float): Heatmap target minimum overlap of cls branch.
            Defaults to 0.8.
        more_pos_thresh (float): The filtering threshold when the cls branch
            adds more positive samples. Defaults to 0.2.
        more_pos_topk (int): The maximum number of additional positive samples
            added to each gt. Defaults to 9.
        soft_weight_on_reg (bool): Whether to use the soft target of the
            cls branch as the soft weight of the bbox branch.
            Defaults to False.
        loss_cls (:obj:`ConfigDict` or dict): Config of cls loss. Defaults to
            dict(type='GaussianFocalLoss', loss_weight=1.0)
        loss_bbox (:obj:`ConfigDict` or dict): Config of bbox loss. Defaults to
             dict(type='GIoULoss', loss_weight=2.0).
        norm_cfg (:obj:`ConfigDict` or dict, optional): dictionary to construct
            and config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config.
            Unused in CenterNet. Reserved for compatibility with
            SingleStageDetector.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config
            of CenterNet.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((0, 80), (64, 160), (128, 320),
                                              (256, 640), (512, INF)),
                 hm_min_radius: int = 4,
                 hm_min_overlap: float = 0.8,
                 more_pos_thresh: float = 0.2,
                 more_pos_topk: int = 9,
                 soft_weight_on_reg: bool = False,
                 loss_cls: ConfigType = dict(
                     type='GaussianFocalLoss',
                     pos_weight=0.25,
                     neg_weight=0.75,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='GIoULoss', loss_weight=2.0),
                 norm_cfg: OptConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        self.soft_weight_on_reg = soft_weight_on_reg
        self.hm_min_radius = hm_min_radius
        self.more_pos_thresh = more_pos_thresh
        self.more_pos_topk = more_pos_topk
        self.delta = (1 - hm_min_overlap) / (1 + hm_min_overlap)
        self.sigmoid_clamp = 0.0001

        # GaussianFocalLoss must be sigmoid mode
        self.use_sigmoid_cls = True
        self.cls_out_channels = num_classes

        self.regress_ranges = regress_ranges
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self._init_reg_convs()
        self._init_predictor()

    def _init_predictor(self) -> None:
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is 4.
        """
        return multi_apply(self.forward_single, x, self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor]:
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps.

        Returns:
            tuple: scores for each class, bbox predictions of
            input feature maps.
        """
        feat = self.reg_convs(x)
        cls_score = self.conv_cls(feat)
        bbox_pred = self.reg_convs(feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        # bbox_pred needed for gradient computation has been modified
        # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
        # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
        bbox_pred = bbox_pred.clamp(min=0)
        if not self.training:
            bbox_pred *= stride
        return cls_score, bbox_pred
