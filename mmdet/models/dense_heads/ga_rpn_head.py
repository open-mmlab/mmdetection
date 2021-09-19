# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.ops import nms

from ..builder import HEADS
from .guided_anchor_head import GuidedAnchorHead


@HEADS.register_module()
class GARPNHead(GuidedAnchorHead):
    """Guided-Anchor-based RPN head."""

    def __init__(self,
                 in_channels,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_loc',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(GARPNHead, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        super(GARPNHead, self)._init_layers()

    def forward_single(self, x):
        """Forward feature of a single scale level."""

        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        (cls_score, bbox_pred, shape_pred,
         loc_pred) = super(GARPNHead, self).forward_single(x)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             shape_preds,
             loc_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        losses = super(GARPNHead, self).loss(
            cls_scores,
            bbox_preds,
            shape_preds,
            loc_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'],
            loss_rpn_bbox=losses['loss_bbox'],
            loss_anchor_shape=losses['loss_shape'],
            loss_anchor_loc=losses['loss_loc'])

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           mlvl_masks,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg

        cfg = copy.deepcopy(cfg)

        # deprecate arguments warning
        if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
            warnings.warn(
                'In rpn_proposal or test_cfg, '
                'nms_thr has been moved to a dict named nms as '
                'iou_threshold, max_num has been renamed as max_per_img, '
                'name of original arguments and the way to specify '
                'iou_threshold of NMS will be deprecated.')
        if 'nms' not in cfg:
            cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
        if 'max_num' in cfg:
            if 'max_per_img' in cfg:
                assert cfg.max_num == cfg.max_per_img, f'You ' \
                    f'set max_num and max_per_img at the same time, ' \
                    f'but get {cfg.max_num} ' \
                    f'and {cfg.max_per_img} respectively' \
                    'Please delete max_num which will be deprecated.'
            else:
                cfg.max_per_img = cfg.max_num
        if 'nms_thr' in cfg:
            assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set ' \
                f'iou_threshold in nms and ' \
                f'nms_thr at the same time, but get ' \
                f'{cfg.nms.iou_threshold} and {cfg.nms_thr}' \
                f' respectively. Please delete the ' \
                f'nms_thr which will be deprecated.'

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
                anchors, rpn_bbox_pred, max_shape=img_shape)
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
        return proposals
