# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weighted_loss


@weighted_loss
def knowledge_distillation_kl_div_loss(pred: Tensor,
                                       soft_label: Tensor,
                                       T: int,
                                       detach_target: bool = True) -> Tensor:
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
            T * T)

    return kd_loss


@MODELS.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 T: int = 10) -> None:
        super().__init__()
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred: Tensor,
                soft_label: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_kd
