import torch.nn as nn
from mmdet.core import (weighted_cross_entropy, weighted_binary_cross_entropy,
                        mask_cross_entropy)

from ..registry import LOSSES


@LOSSES.register_module
class CrossEntropyLoss(nn.Module):

    def __init__(self, use_sigmoid=False, use_mask=False, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = weighted_binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = weighted_cross_entropy

    def forward(self, cls_score, label, label_weight, *args, **kwargs):
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, label, label_weight, *args, **kwargs)
        return loss_cls
