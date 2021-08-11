import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def binary_heatmap_focal_loss(pred, target_pos_inds, alpha=2.0, gamma=4.0, beta=0.25 ,sigmoid_clamp=0.0001, ignore_high_fp=0.85):
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
    pred = torch.clamp(pred.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
    targets = target_pos_inds[0]
    pos_inds = target_pos_inds[1]
    neg_weights = torch.pow(1 - targets, gamma)
    pos_pred = pred[pos_inds]
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, alpha)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights

    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss

    pos_loss = - pos_loss.sum()
    neg_loss = - neg_loss.sum()

    if beta >= 0:
        pos_loss = beta * pos_loss
        neg_loss = (1 - beta) * neg_loss

    loss = 0.5*pos_loss + 0.5*neg_loss

    return loss


@LOSSES.register_module()
class HeatmapBinaryFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.

    More details can be found in the `paper
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
                 alpha=2.0,
                 gamma=4.0,
                 beta=0.25,
                 sigmoid_clamp=1e-4,
                 ignore_high_fp=-1,
                 reduction='mean',
                 loss_weight=1.0):
        super(HeatmapBinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigmoid_clamp = sigmoid_clamp
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
        target_pos_inds = [target, pos_inds]
        loss_reg = self.loss_weight * binary_heatmap_focal_loss(
            pred,
            target_pos_inds,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            sigmoid_clamp=self.sigmoid_clamp,
            ignore_high_fp=self.ignore_high_fp,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg
