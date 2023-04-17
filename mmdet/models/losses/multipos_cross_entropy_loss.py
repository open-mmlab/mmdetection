# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weight_reduce_loss


@MODELS.register_module()
class MultiPosCrossEntropyLoss(BaseModule):
    """multi-positive targets cross entropy loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0):
        super(MultiPosCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def multi_pos_cross_entropy(self,
                                pred: Tensor,
                                label: Tensor,
                                weight: Optional[Tensor] = None,
                                reduction: str = 'mean',
                                avg_factor: Optional[float] = None) -> Tensor:
        """Multi-positive targets cross entropy loss.

        Args:
            pred (torch.Tensor): The prediction.
            label (torch.Tensor): The assigned label of the prediction.
            weight (torch.Tensor): The element-wise weight.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Average factor when computing
                the mean of losses.

        Returns:
            torch.Tensor: Calculated loss
        """

        pos_inds = (label >= 1)
        neg_inds = (label == 0)
        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

        _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
        _neg_expand = pred_neg.repeat(1, pred.shape[1])

        x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1),
                                    'constant', 0)
        loss = torch.logsumexp(x, dim=1)

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss

    def forward(self,
                cls_score: Tensor,
                label: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The classification score.
            label (torch.Tensor): The assigned label of the prediction.
            weight (torch.Tensor): The element-wise weight.
            avg_factor (float): Average factor when computing
                the mean of losses.
            reduction_override (str): Same as built-in losses of PyTorch.

        Returns:
            torch.Tensor: Calculated loss
        """
        assert cls_score.size() == label.size()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.multi_pos_cross_entropy(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls
