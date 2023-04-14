if '_base_':
    from ..faster_rcnn.faster_rcnn_r50_fpn_1x_coco import *
from mmdet.models.roi_heads.double_roi_head import DoubleHeadRoIHead
from mmdet.models.roi_heads.bbox_heads.double_bbox_head import DoubleConvFCBBoxHead
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss

model.merge(
    dict(
        roi_head=dict(
            type=DoubleHeadRoIHead,
            reg_roi_scale_factor=1.3,
            bbox_head=dict(
                _delete_=True,
                type=DoubleConvFCBBoxHead,
                num_convs=4,
                num_fcs=2,
                in_channels=256,
                conv_out_channels=1024,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type=DeltaXYWHBBoxCoder,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=2.0),
                loss_bbox=dict(type=SmoothL1Loss, beta=1.0,
                               loss_weight=2.0)))))
