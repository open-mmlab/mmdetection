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
        target_pos_inds (list[torch.Tensor]): The learning target and
            positive indices.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        beta (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        sigmoid_clamp (float, optional): A value used to determine
                clamp range.
        ignore_high_fp (float, optional): A threshold to ignore sample
            points with high positive scores when calculating negative
            loss.
    """
    pred = torch.clamp(
    	pred.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)
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

    loss = pos_loss + neg_loss

    return loss


@LOSSES.register_module()
class HeatmapBinaryFocalLoss(nn.Module):
    """A Gaussian heatmap focal loss calculation, it use a small portion of
    points as positive sample points.

        `Probabilistic two-stage detection
        <https://arxiv.org/abs/2103.07461>`_
        `Focal Loss
        <https://arxiv.org/abs/1708.02002>`_

    Args:
        alpha (float, optional): The alpha for calculating the positive
            sample points loss modulating factor. Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the negative
            sample points loss modulating factor. Defaults to 4.0.
        beta (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        sigmoid_clamp (float, optional): A value used to determine
            clamp range.
        ignore_high_fp (float, optional): A threshold to ignore sample
            points with high positive scores when calculating negative
            loss.
        reduction (string, optional): The method used to reduce the
            loss into a scalar. Defaults to 'sum' for heatmap loss.
        loss_weight (str, optional): Weight of agn_hm loss.
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
            pos_inds (torch.Tensor): Indices of positive sample points.
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
        loss_agn_hm = self.loss_weight * binary_heatmap_focal_loss(
            pred,
            target_pos_inds,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            sigmoid_clamp=self.sigmoid_clamp,
            ignore_high_fp=self.ignore_high_fp,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_agn_hm
