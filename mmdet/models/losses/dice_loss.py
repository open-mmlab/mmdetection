import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        assert pred.shape == target.shape
        I = torch.sum(pred * target)
        U = torch.sum(pred) + torch.sum(target)
        loss = self.loss_weight * (1 - 2.0 * I / U) if U > 0 else U
        return loss
