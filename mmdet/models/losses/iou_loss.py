import torch.nn as nn
from mmdet.core import weighted_iou_loss

from ..registry import LOSSES


@LOSSES.register_module
class IoULoss(nn.Module):

    def __init__(self, style='naive', beta=0.2, eps=1e-3, loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.style = style
        self.beta = beta
        self.eps = eps
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        loss = self.loss_weight * weighted_iou_loss(
            pred,
            target,
            weight,
            style=self.style,
            beta=self.beta,
            eps=self.eps,
            *args,
            **kwargs)
        return loss
