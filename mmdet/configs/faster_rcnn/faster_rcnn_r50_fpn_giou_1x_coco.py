if '_base_':
    from .faster_rcnn_r50_fpn_1x_coco import *
from mmdet.models.losses.iou_loss import GIoULoss

model.merge(
    dict(
        roi_head=dict(
            bbox_head=dict(
                reg_decoded_bbox=True,
                loss_bbox=dict(type=GIoULoss, loss_weight=10.0)))))
