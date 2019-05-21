from mmdet.core import weighted_sigmoid_focal_loss

from .cross_entropy_loss import CrossEntropyLoss
from ..registry import LOSSES


@LOSSES.register_module
class FocalLoss(CrossEntropyLoss):

    def __init__(self, gamma=2.0, alpha=0.25, *args, **kwargs):
        super(FocalLoss, self).__init__(*args, **kwargs)
        assert self.use_sigmoid is True, 'Only sigmoid focaloss supported now.'
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
            raise NotImplementedError()
        return loss_cls
