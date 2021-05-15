import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss



@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 loss_weight=1.0):
        """``_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        if self.use_sigmoid:
            pred = F.sigmoid(pred)
        input = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1).float()

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        loss_cls = self.loss_weight * (1 - d)

        return loss_cls
