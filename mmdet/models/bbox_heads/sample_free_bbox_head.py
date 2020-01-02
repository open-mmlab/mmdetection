from mmdet.core import force_fp32
from ..anchor_heads.sample_free_head import GuidedLoss
from ..losses import accuracy
from ..registry import HEADS
from .convfc_bbox_head import SharedFCBBoxHead


@HEADS.register_module
class SampleFreeBBoxHead(SharedFCBBoxHead):
    def __init__(self, loss_cls_scale, **kwargs):
        super(SampleFreeBBoxHead, self).__init__(**kwargs)
        self.guided_loss = GuidedLoss(loss_cls_scale)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """almost the same with origin loss, here we just modify the avg_factor.
        when use down-sample, the avg_factor of bbox loss is the num of all
        sample(512 per image). if we use sample-free, due to we ont apply
        sample, so the avg_factor will be larger, bbox loss will be smaller,
        here we set avg_factor = None to rescale the loss
        """
        losses = dict()
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=None,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0),
                                                   4)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                                   4)[pos_inds,
                                                      labels[pos_inds]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds],
                    bbox_weights[pos_inds],
                    avg_factor=None,
                    reduction_override=reduction_override)

            losses['loss_cls'] = self.guided_loss(
                losses['loss_bbox'], losses['loss_cls'])

        return losses
