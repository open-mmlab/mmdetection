import torch

from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from .builder import MATCH_COST


@MATCH_COST.register_module()
class BBoxL1Cost(object):

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes, factor):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            factor (Tensor): Image size(whwh) tensor. Shape [4].
        """
        gt_bboxes_normalized = gt_bboxes / factor
        # MODYFY: xyxy p1-norm in sparse-RCNN CODE which need to review
        bbox_cost = torch.cdist(
            bbox_pred, bbox_xyxy_to_cxcywh(gt_bboxes_normalized), p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class ClsFocalCost(object):

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class ClsSoftmaxCost(object):

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be ommitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class IoUBasedCost(object):

    def __init__(self,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 iou_mode='giou',
                 weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = self.iou_calculator(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so ommitted.
        iou_cost = -overlaps
        return iou_cost * self.weight
