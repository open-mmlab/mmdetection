import torch
import torch.nn as nn

from mmdet.models.losses.utils import weighted_loss
from mmdet.registry import MODELS


@weighted_loss
def cn2_gaussian_focal_loss(pred,
                            gaussian_target,
                            pos_inds=None,
                            alpha: float = -1,
                            beta: float = 4,
                            gamma: float = 2,
                            sigmoid_clamp: float = 1e-4,
                            ignore_high_fp: float = -1.):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    pred = torch.clamp(
        pred.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
    neg_weights = torch.pow(1 - gaussian_target, beta)
    pos_pred = pred[pos_inds]  # N
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, gamma)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights
    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    pos_loss = -pos_loss.sum()
    neg_loss = -neg_loss.sum()

    if alpha >= 0:
        pos_loss = alpha * pos_loss
        neg_loss = (1 - alpha) * neg_loss

    return pos_loss + neg_loss


@MODELS.register_module()
class CN2GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss. More details can be found
    in the `paper.

    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.
    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha: float = -1,
                 beta: float = 4,
                 gamma: float = 2,
                 sigmoid_clamp: float = 1e-4,
                 ignore_high_fp: float = -1.,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigmoid_clmap = sigmoid_clamp
        self.ignore_high_fp = ignore_high_fp
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                pos_inds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * cn2_gaussian_focal_loss(
            pred,
            target,
            weight,
            pos_inds=pos_inds,
            alpha=self.alpha,
            gamma=self.gamma,
            ignore_high_fp=self.ignore_high_fp,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg
