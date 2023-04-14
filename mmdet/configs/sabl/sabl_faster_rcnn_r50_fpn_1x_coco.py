if '_base_':
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.roi_heads.bbox_heads.sabl_head import SABLHead
from mmdet.models.task_modules.coders.bucketing_bbox_coder import BucketingBBoxCoder
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss, CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss

model.merge(
    dict(
        roi_head=dict(
            bbox_head=dict(
                _delete_=True,
                type=SABLHead,
                num_classes=80,
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                reg_feat_up_ratio=2,
                reg_pre_kernel=3,
                reg_post_kernel=3,
                reg_pre_num=2,
                reg_post_num=1,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                norm_cfg=None,
                bbox_coder=dict(
                    type=BucketingBBoxCoder, num_buckets=14, scale_factor=1.7),
                loss_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0),
                loss_bbox_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=True, loss_weight=1.0),
                loss_bbox_reg=dict(
                    type=SmoothL1Loss, beta=0.1, loss_weight=1.0)))))
