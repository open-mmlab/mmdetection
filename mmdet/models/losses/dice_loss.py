import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def dice_loss(pred, target):
    x = pred
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x**2.0).sum(dim=1) + (target**2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_dice = self.loss_weight * dice_loss(
            pred, target, reduction=reduction, avg_factor=avg_factor)
        return loss_dice
