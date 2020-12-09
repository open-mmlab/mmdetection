# Based off of github.com/rwightman/efficientdet-pytorch

import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss

@weighted_loss
def huber_loss(pred, target, delta):
    diff = abs(pred - target)
    quadratic = torch.clamp(diff, max=delta)
    loss = 0.5 * quadratic.pow(2) + delta * (diff - quadratic)
    return loss

@LOSSES.register_module()
class HuberLoss(nn.Module):
    def __init__(self, delta, loss_weight=None, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,
                input,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        loss = huber_loss(
            input, target, weight, delta=self.delta, reduction=reduction, avg_factor=avg_factor)

        if self.loss_weight is not None:
            loss *= self.loss_weight

        return loss

