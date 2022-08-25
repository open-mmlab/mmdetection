# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from .guided_anchor_head import GuidedAnchorHead


@MODELS.register_module()
class GARPNHead(GuidedAnchorHead):
    """Guided-Anchor-based RPN head."""

    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_loc',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        super(GARPNHead, self)._init_layers()

    def forward_single(self, x: Tensor) -> Tuple[Tensor]:
        """Forward feature of a single scale level."""

        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        (cls_score, bbox_pred, shape_pred,
         loc_pred) = super().forward_single(x)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            shape_preds: List[Tensor],
            loc_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            shape_preds (list[Tensor]): shape predictions for each scale
                level with shape (N, 1, H, W).
            loc_preds (list[Tensor]): location predictions for each scale
                level with shape (N, num_anchors * 2, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            shape_preds,
            loc_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'],
            loss_rpn_bbox=losses['loss_bbox'],
            loss_anchor_shape=losses['loss_shape'],
            loss_anchor_loc=losses['loss_loc'])

    def _predict_by_feat_single(self,
                                cls_scores: List[Tensor],
                                bbox_preds: List[Tensor],
                                mlvl_anchors: List[Tensor],
                                mlvl_masks: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = False) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            mlvl_anchors (list[Tensor]): Each element in the list is
                the anchors of a single level in feature pyramid. it has
                shape (num_priors, 4).
            mlvl_masks (list[Tensor]): Each element in the list is location
                masks of a single level.
            img_meta (dict): Image meta info.
            cfg (:obj:`ConfigDict` or dict): Test / postprocessing
                configuration, if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4), the last
              dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        assert cfg.nms.get('type', 'nms') == 'nms', 'GARPNHead only support ' \
            'naive nms.'

        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]
            mask = mlvl_masks[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = rpn_cls_score.softmax(dim=1)[:, :-1]
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            scores = scores[mask]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1,
                                                                   4)[mask, :]
            if scores.dim() == 0:
                rpn_bbox_pred = rpn_bbox_pred.unsqueeze(0)
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            # get proposals w.r.t. anchors and rpn_bbox_pred
            proposals = self.bbox_coder.decode(
                anchors, rpn_bbox_pred, max_shape=img_meta['img_shape'])
            # filter out too small bboxes
            if cfg.min_bbox_size >= 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
                if not valid_mask.all():
                    proposals = proposals[valid_mask]
                    scores = scores[valid_mask]

            # NMS in current level
            proposals, _ = nms(proposals, scores, cfg.nms.iou_threshold)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.get('nms_across_levels', False):
            # NMS across multi levels
            proposals, _ = nms(proposals[:, :4], proposals[:, -1],
                               cfg.nms.iou_threshold)
            proposals = proposals[:cfg.max_per_img, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_per_img, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]

        bboxes = proposals[:, :-1]
        scores = proposals[:, -1]
        if rescale:
            assert img_meta.get('scale_factor') is not None
            bboxes /= bboxes.new_tensor(img_meta['scale_factor']).repeat(
                (1, 2))

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = scores
        results.labels = scores.new_zeros(scores.size(0), dtype=torch.long)
        return results
