if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.cityscapes_instance import *
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_1x import *
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

model.merge(
    dict(
        backbone=dict(init_cfg=None),
        roi_head=dict(
            bbox_head=dict(
                type=Shared2FCBBoxHead,
                num_classes=8,
                loss_bbox=dict(type=SmoothL1Loss, beta=1.0, loss_weight=1.0)),
            mask_head=dict(num_classes=8))))

# optimizer
# lr is set for a batch size of 8
optim_wrapper.merge(dict(optimizer=dict(lr=0.01)))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=8,
        by_epoch=True,
        # [7] yields higher performance than [6]
        milestones=[7],
        gamma=0.1)
]

# actual epoch = 8 * 8 = 64
train_cfg.merge(dict(max_epochs=8))

# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
# TODO: support auto scaling lr
# auto_scale_lr = dict(base_batch_size=8)
