import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def dice_loss(pred, target, eps=1e-5):
    """Dice loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        eps (int | float, optional): A value added to the denominator for
            numerical stability. Default 1e-5.

    Returns:
        torch.Tensor: The calculated loss
    """
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = (pred**2.0).sum(dim=1) + (target**2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, eps=1e-5):
        """Implementation of Dice Loss.

        The loss can be described as:
        .. math::
            \text{loss}(pred, target) = 1 -
                    \frac{2|pred \bigcap target|}{|pred|+|target|}

        Args:
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-5.
        """
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_dice = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_dice
