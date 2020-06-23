import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.ops import batched_nms
from mmdet.core.post_processing import multiclass_nms
from mmdet.core.bbox.coder.delta_xywh_bbox_coder import delta2bbox
from mmdet.core.utils.misc import topk
from ..builder import HEADS
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin


@HEADS.register_module()
class RPNHead(RPNTestMixin, AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(
            1, in_channels, background_label=0, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            nms_pre = int(cfg.nms_pre)
            scores, topk_inds = topk(scores, nms_pre)
            rpn_bbox_pred = rpn_bbox_pred[topk_inds]
            anchors = anchors[topk_inds]
            proposals = self.bbox_coder.decode(
                    anchors, rpn_bbox_pred, max_shape=img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals, _ = multiclass_nms(proposals, scores.unsqueeze(1), 0.0,
                                          {
                                              'type': 'nms',
                                              'iou_thr': cfg.nms_thr
                                          }, cfg.nms_post)
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = multiclass_nms(proposals[:, :4],
                                          proposals[:, 4].unsqueeze(1), 0.0, {
                                              'type': 'nms',
                                              'iou_thr': cfg.nms_thr
                                          }, cfg.max_num)
        else:
            scores = proposals[:, 4]
            _, topk_inds = topk(scores, cfg.max_num)
            proposals = proposals[topk_inds]
        return proposals
