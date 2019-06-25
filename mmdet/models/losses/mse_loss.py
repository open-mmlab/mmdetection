import torch.nn as nn
import torch.nn.functional as F

from .utils import weighted_loss
from ..registry import LOSSES

mse_loss = weighted_loss(F.mse_loss)


@LOSSES.register_module
class MSELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)
        return loss
