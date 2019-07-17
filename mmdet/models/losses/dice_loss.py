import torch
import torch.nn as nn

from ..registry import LOSSES


def dice_loss(pred, target, label, reduction='mean', avg_factor=None):
    """
    using for mask loss
    :param pred:   [n_rois, n_cls, h, w]
    :param target: [n_rois, h, w]
    :param label:  [n_rois]
    :return:
    """
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    pred_slice = torch.sigmoid(pred_slice)

    ovr = (pred_slice * target).sum(dim=[1, 2])
    union = (pred_slice + target).sum(dim=[1, 2]).clamp(1e-10)
    return (1 - (2*ovr/union).sum()/num_rois)[None]


@LOSSES.register_module
class DiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, weight=None, avg_factor=None,
                **kwargs):
        return self.loss_weight * dice_loss(cls_score, label, weight)

