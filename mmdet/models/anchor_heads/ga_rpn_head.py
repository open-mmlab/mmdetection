import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from .guided_anchor_head import GuidedAnchorHead
from ..registry import HEADS


@HEADS.register_module
class GARPNHead(GuidedAnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(GARPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        super(GARPNHead, self)._init_layers()

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        super(GARPNHead, self).init_weights()

    def forward_single(self, x):
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
             cfg,
             gt_bboxes_ignore=None):
        losses = super(GARPNHead, self).loss(
            cls_scores,
            bbox_preds,
            shape_preds,
            loc_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'],
            loss_rpn_reg=losses['loss_reg'],
            loss_rpn_shape=losses['loss_shape'],
            loss_rpn_loc=losses['loss_loc'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          shape_preds,
                          loc_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            shape_pred = shape_preds[idx]
            loc_pred = loc_preds[idx].sigmoid()
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size(
            )[-2:] == shape_pred.size()[-2:] == loc_pred.size()[-2:]
            loc_mask = loc_pred[0] >= cfg.loc_filter_thr
            mask = loc_mask[..., None].expand(
                loc_mask.size(0), loc_mask.size(1), self.num_anchors)
            mask = mask.contiguous().view(-1)
            mask_inds = mask.nonzero()
            if mask_inds.numel() == 0:
                continue
            else:
                mask_inds = mask_inds.squeeze()
            anchors = mlvl_anchors[idx][mask_inds]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.cls_sigmoid_loss:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            scores = scores[mask_inds]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(
                -1, 4)[mask_inds, :]
            shape_pred = shape_pred.permute(1, 2, 0).reshape(-1,
                                                             2)[mask_inds, :]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
                shape_pred = shape_pred[topk_inds, :]
            anchor_deltas = shape_pred.new_full((shape_pred.size(0), 4), 0)
            anchor_deltas[:, 2:] = shape_pred
            pred_anchors = delta2bbox(anchors, anchor_deltas,
                                      self.anchoring_means,
                                      self.anchoring_stds, img_shape)
            proposals = delta2bbox(pred_anchors, rpn_bbox_pred,
                                   self.target_means, self.target_stds,
                                   img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
