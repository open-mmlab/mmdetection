import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weighted_loss

mse_loss = weighted_loss(F.mse_loss)#这里不用再次定义mse_loss了，所以就没有@weight_loss了，直接将函数放到里面就可以了


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
