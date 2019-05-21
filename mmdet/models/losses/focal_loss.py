import torch.nn as nn
from mmdet.core import weighted_sigmoid_focal_loss

from ..registry import LOSSES


@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 loss_weight=1.0,
                 gamma=2.0,
                 alpha=0.25):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focaloss supported now.'
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.cls_criterion = weighted_sigmoid_focal_loss

    def forward(self, cls_score, label, label_weight, *args, **kwargs):
        if self.use_sigmoid:
            loss_cls = self.loss_weight * self.cls_criterion(
                cls_score,
                label,
                label_weight,
                gamma=self.gamma,
                alpha=self.alpha,
                *args,
                **kwargs)
        else:
            raise NotImplementedError
        return loss_cls
