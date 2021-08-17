import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-3):
        """`Dice Loss, which is proposed in
        `V-Net: Fully Convolutional Neural Networks for Volumetric
         Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Default: 'mean'.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Default: 1e-3.
        """
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None,
                has_acted=False):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has shape (n, *)
            target (torch.Tensor): The learning label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            has_acted (bool): Has been activated outside, this will disable
                the in

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if not has_acted:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        input = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1).float()

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + self.eps
        c = torch.sum(target * target, 1) + self.eps
        d = (2 * a) / (b + c)
        loss = self.loss_weight * (1 - d)
        if weight is not None:
            assert weight.ndim == loss.ndim
            assert len(weight) == len(pred)

        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss
