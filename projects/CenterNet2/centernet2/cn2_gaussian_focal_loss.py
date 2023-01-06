# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss


def cn2_gaussian_focal_loss_with_pos_inds(
        pred: Tensor,
        gaussian_target: Tensor,
        pos_inds: Tensor,
        pos_labels: Tensor,
        alpha: float = 2.0,
        gamma: float = 4.0,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
	ignore_high_fp: float = -1.,
        reduction: str = 'mean',
        avg_factor: Optional[Union[int, float]] = None) -> Tensor:
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Note: The index with a value of 1 in ``gaussian_target`` in the
    ``gaussian_focal_loss`` function is a positive sample, but in
    ``gaussian_focal_loss_with_pos_inds`` the positive sample is passed
    in through the ``pos_inds`` parameter.

    Args:
        pred (torch.Tensor): The prediction. The shape is (N, num_classes).
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution. The shape is (N, num_classes).
        pos_inds (torch.Tensor): The positive sample index.
            The shape is (M, ).
        pos_labels (torch.Tensor): The label corresponding to the positive
            sample index. The shape is (M, ).
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
        reduction (str): Options are "none", "mean" and "sum".
            Defaults to 'mean`.
        avg_factor (int, float, optional): Average factor that is used to
            average the loss. Defaults to None.
    """
    eps = 1e-12
    neg_weights = (1 - gaussian_target).pow(gamma)

    pos_pred_pix = pred[pos_inds]
    pos_pred = pos_pred_pix.gather(1, pos_labels.unsqueeze(1))
    pos_loss = -(pos_pred + eps).log() * (1 - pos_pred).pow(alpha)
    pos_loss = weight_reduce_loss(pos_loss, None, reduction, avg_factor)
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights

    if ignore_high_fp > 0:
        not_high_fp = (pred < ignore_high_fp).float()
        neg_loss = not_high_fp * neg_loss
	
    neg_loss = weight_reduce_loss(neg_loss, None, reduction, avg_factor)

    return pos_weight * pos_loss + neg_weight * neg_loss


@MODELS.register_module()
class CN2GaussianFocalLoss(nn.Module):
    """CenterNet2 GaussianFocalLoss is a variant of focal loss.

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
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """

    def __init__(self,
                 alpha: float = 2.0,
                 gamma: float = 4.0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0,
		         ignore_high_fp: float = -1.) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.ignore_high_fp = ignore_high_fp

    def forward(self,
                pred: Tensor,
                target: Tensor,
                pos_inds: Tensor,
                pos_labels: Tensor = None,
                avg_factor: Optional[Union[int, float]] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction. The shape is (N, num_classes).
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution. The shape is (N, num_classes).
            pos_inds (torch.Tensor): The positive sample index.
            pos_labels (torch.Tensor): The label corresponding to the positive
                sample index.
            avg_factor (int, float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert pos_labels is not None
        # Only used by centernet update version
        loss_reg = self.loss_weight * cn2_gaussian_focal_loss_with_pos_inds(
            pred,
            target,
            pos_inds,
            pos_labels,
            alpha=self.alpha,
            gamma=self.gamma,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_reg
