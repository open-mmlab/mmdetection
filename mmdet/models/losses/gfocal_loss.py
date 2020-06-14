import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def quality_focal_loss(pred,
                       label,
                       score,
                       weight=None,
                       beta=2.0,
                       reduction='mean',
                       avg_factor=None):
    # all goes to 0
    pred_sigmoid = pred.sigmoid()
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * pt.pow(beta)  # pt = pt.abs()
    # find positive positions and quality labels
    label = label - 1
    pos = (label >= 0).nonzero().squeeze(1)
    a = pos
    b = label[pos].long()
    # positive goes to bbox quality, e.g., IoU target
    pt = score[a] - pred_sigmoid[a, b]
    loss[a, b] = F.binary_cross_entropy_with_logits(
        pred[a, b], score[a], reduction='none') * pt.abs().pow(beta)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def distribution_focal_loss(pred,
                            label,
                            weight=None,
                            reduction='mean',
                            avg_factor=None):
    disl = label.long()
    disr = disl + 1
    wl = disr.float() - label
    wr = label - disl.float()
    loss = F.cross_entropy(pred, disl, reduction='none') * wl \
        + F.cross_entropy(pred, disr, reduction='none') * wr
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module
class QualityFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                score,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                score,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


@LOSSES.register_module
class DistributionFocalLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_cls
