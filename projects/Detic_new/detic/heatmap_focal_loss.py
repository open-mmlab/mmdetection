# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS


# support class-agnostic heatmap_focal_loss
def heatmap_focal_loss_with_pos_inds(
        pred: Tensor,
        targets: Tensor,
        pos_inds: Tensor,
        alpha: float = 2.0,
        beta: float = 4.0,
        gamma: float = 4.0,
        sigmoid_clamp: float = 1e-4,
        ignore_high_fp: float = -1.0,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
        avg_factor: Optional[Union[int, float]] = None) -> Tensor:

    pred = torch.clamp(
        pred.sigmoid_(), min=sigmoid_clamp, max=1 - sigmoid_clamp)

    neg_weights = torch.pow(1 - targets, beta)

    pos_pred = pred[pos_inds]
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

    pos_loss = pos_weight * pos_loss / avg_factor
    neg_loss = neg_weight * neg_loss / avg_factor

    return pos_loss, neg_loss


@MODELS.register_module()
class HeatmapFocalLoss(nn.Module):
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
        pos_weight(float): Positive sample loss weight. Defaults to 1.0.
        neg_weight(float): Negative sample loss weight. Defaults to 1.0.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        gamma: float = 4.0,
        sigmoid_clamp: float = 1e-4,
        ignore_high_fp: float = -1.0,
        loss_weight: float = 1.0,
        pos_weight: float = 1.0,
        neg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigmoid_clamp = sigmoid_clamp
        self.ignore_high_fp = ignore_high_fp
        self.loss_weight = loss_weight
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                pos_inds: Optional[Tensor] = None,
                avg_factor: Optional[Union[int, float]] = None) -> Tensor:
        """Forward function.

        If you want to manually determine which positions are
        positive samples, you can set the pos_index and pos_label
        parameter. Currently, only the CenterNet update version uses
        the parameter.

        Args:
            pred (torch.Tensor): The prediction. The shape is (N, num_classes).
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution. The shape is (N, num_classes).
            pos_inds (torch.Tensor): The positive sample index.
                Defaults to None.
            pos_labels (torch.Tensor): The label corresponding to the positive
                sample index. Defaults to None.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """

        pos_loss, neg_loss = heatmap_focal_loss_with_pos_inds(
            pred,
            target,
            pos_inds,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            sigmoid_clamp=self.sigmoid_clamp,
            ignore_high_fp=self.ignore_high_fp,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            avg_factor=avg_factor)
        return pos_loss, neg_loss
