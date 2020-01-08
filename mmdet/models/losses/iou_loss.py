import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def iou_loss(pred, target, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss


@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164.

    Args:
        pred (tensor): Predicted bboxes.
        target (tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0] + 1
    pred_h = pred[:, 3] - pred[:, 1] + 1
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0] + 1
        target_h = target[:, 3] - target[:, 1] + 1

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).view(loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    return loss


@weighted_loss
def giou_loss(pred, target, eps=1e-6):
    """
    Generalized Intersection over Union: A Metric and A Loss for
    Bounding Box Regression
    https://arxiv.org/abs/1902.09630

    code refer to:
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """

    pred_x1 = pred[:, 0]
    pred_y1 = pred[:, 1]
    pred_x2 = pred[:, 2]
    pred_y2 = pred[:, 3]
    pred_x2 = torch.max(pred_x1, pred_x2)
    pred_y2 = torch.max(pred_y1, pred_y2)
    pred_area = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)

    target_x1 = target[:, 0]
    target_y1 = target[:, 1]
    target_x2 = target[:, 2]
    target_y2 = target[:, 3]
    target_area = (target_x2 - target_x1 + 1) * (target_y2 - target_y1 + 1)

    x1_intersect = torch.max(pred_x1, target_x1)
    y1_intersect = torch.max(pred_y1, target_y1)
    x2_intersect = torch.min(pred_x2, target_x2)
    y2_intersect = torch.min(pred_y2, target_y2)
    area_intersect = torch.zeros(pred_x1.size()).to(pred)
    mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
    area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask] + 1) * (
        y2_intersect[mask] - y1_intersect[mask] + 1)

    x1_enclosing = torch.min(pred_x1, target_x1)
    y1_enclosing = torch.min(pred_y1, target_y1)
    x2_enclosing = torch.max(pred_x2, target_x2)
    y2_enclosing = torch.max(pred_y2, target_y2)
    area_enclosing = (x2_enclosing - x1_enclosing +
                      1) * (y2_enclosing - y1_enclosing + 1) + 1e-7

    area_union = pred_area + target_area - area_intersect + 1e-7
    ious = area_intersect / area_union
    gious = ious - (area_enclosing - area_union) / area_enclosing
    loss = 1 - gious

    return loss


@LOSSES.register_module
class IoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module
class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=1e-3, reduction='mean', loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * bounded_iou_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module
class GIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
