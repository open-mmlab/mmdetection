import torch.nn as nn

from ..builder import LOSSES


def gaussian_focal_loss(pred, target, weight=None, alpha=2.0, gamma=4.0):
    eps = 1e-12
    pos_weights = target.eq(1)
    neg_weights = (1 - target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights

    if weight is not None:
        pos_loss *= weight
        neg_loss *= weight

    pos_num = pos_weights.sum()
    if pos_num < 1:
        loss = neg_loss.sum()
    else:
        loss = (pos_loss + neg_loss).sum() / pos_num
    return loss


@LOSSES.register_module()
class GaussianFocalLoss(nn.Module):
    """ GaussianFocalLoss is a variant of focal loss.

    Please refer to https://arxiv.org/abs/1808.01244 for more details.
    Code is modified from https://github.com/princeton-vl/CornerNet.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negtive samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='none',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None):
        loss_reg = self.loss_weight * gaussian_focal_loss(
            pred, target, weight, alpha=self.alpha, gamma=self.gamma)
        return loss_reg
