import mmcv
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def knowledge_distillation_loss(pred, soft_label, T, detach_target=True):
    r"""Knowledge distillation is a model compression method introduced
    by Hinto in 2015, which transfer the knowledge from a large model to a
    small one without loss of validity.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
    pred (torch.Tensor): Predicted general distribution of bounding boxes
        (before softmax) with shape (N, n+1), n is the max value of the
        integral set `{0, ..., n}` in paper.
    soft_label (torch.Tensor): Target soft label learned from teacher
         with shape (N, n+1).
    T (int): Temperature for distillation.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """

    target = F.softmax(soft_label / T, dim=1)
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (
            T * T)

    return kd_loss


@LOSSES.register_module()
class LocalizationDistillationLoss(nn.Module):
    r"""LD is the extension of knowledge distillation on localization task,
    which utilizes the learned bbox distributions to transfer the localization
    dark knowledge from teacher to student.
    <https://arxiv.org/abs/2102.12252>`_.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(LocalizationDistillationLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def forward(self,
                pred,
                soft_corners,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted general distribution of bounding
                boxes (before softmax) with shape (N, n+1), n is the max value
                of the integral set `{0, ..., n}` in paper.
            soft_corners (torch.Tensor): Target distance label learned from
                teacher for bounding boxes with shape (N, n+1)
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

        loss_ld = self.loss_weight * knowledge_distillation_loss(
            pred,
            soft_corners,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            T=self.T)

        return loss_ld
