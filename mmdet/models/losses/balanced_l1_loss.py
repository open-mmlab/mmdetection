import numpy as np
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def balanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction='mean'):
    """Calculate balanced L1 loss

    Please see the `Libra R-CNN <https://arxiv.org/pdf/1904.02701.pdf>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, 4).
        target (torch.Tensor): The learning target of the prediction with
            shape (N, 4).
        beta (float): The loss is a piecewise function of prediction and target
            and ``beta`` serves as a threshold for the difference between the
            prediction and target. Defaults to 1.0.
        alpha (float): The denominator ``alpha`` in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The ``gamma`` in the balanced L1 loss.
            Defaults to 1.5.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".

    Returns:
        torch.Tensor: The calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    return loss


@LOSSES.register_module()
class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Args:
        alpha (float): The denominator ``alpha`` in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The ``gamma`` in the balanced L1 loss. Defaults to 1.5.
        beta (float, optional): The loss is a piecewise function of prediction
            and target. ``beta`` serves as a threshold for the difference
            between the prediction and target. Defaults to 1.0.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0

    """

    def __init__(self,
                 alpha=0.5,
                 gamma=1.5,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function of loss

        Args:
            pred (torch.Tensor): The prediction with shape (N, 4).
            target (torch.Tensor): The learning target of the prediction with
                shape (N, 4).
            weight (torch.Tensor, optional): Sample-wise loss weight with
                shape (N, ).
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * balanced_l1_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
