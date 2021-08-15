import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self, use_sigmoid=True, loss_weight=1.0, eps=1e-3):
        """`Dice Loss `<https://arxiv.org/abs/1912.04488>`_ .

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): To avoid deivde zero. Default 1e-3.
        """
        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (torch.Tensor or tuple):
                The prediction,  if torch.Tensor, shape (n, h, w)
                if tuple, each param is torch.Tensor with shape (n, w, h)
            target (torch.Tensor): The learning label of the prediction,
                shape (n, h, w).

        Returns:
            torch.Tensor: The calculated loss
        """
        assert isinstance(pred, torch.Tensor) or isinstance(pred, tuple)
        if isinstance(pred, tuple):
            assert len(pred) == 2
            if self.use_sigmoid:
                pred = F.sigmoid(pred[0]) * F.sigmoid(pred[1])
            else:
                pred = pred[0] + pred[1]
        else:
            if self.use_sigmoid:
                pred = F.sigmoid(pred)
        input = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1).float()

        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1) + 1e-3
        c = torch.sum(target * target, 1) + 1e-3
        d = (2 * a) / (b + c)
        loss_cls = self.loss_weight * (1 - d)

        return loss_cls
