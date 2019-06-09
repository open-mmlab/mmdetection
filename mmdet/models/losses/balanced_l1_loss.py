import torch.nn as nn
from mmdet.core import weighted_balanced_l1_loss

from ..registry import LOSSES


@LOSSES.register_module
class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0, loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        loss_bbox = self.loss_weight * weighted_balanced_l1_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            *args,
            **kwargs)
        return loss_bbox
