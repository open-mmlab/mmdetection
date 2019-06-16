import torch.nn as nn
from mmdet.core import weighted_generalized_iou_loss

from ..registry import LOSSES


@LOSSES.register_module
class GIoULoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        loss = self.loss_weight * weighted_generalized_iou_loss(pred, target, weight, *args, **kwargs)
        return loss