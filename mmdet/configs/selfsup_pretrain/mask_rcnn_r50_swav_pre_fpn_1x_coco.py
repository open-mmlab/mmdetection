if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.coco_instance import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *

model.merge(
    dict(
        backbone=dict(
            frozen_stages=0,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            norm_eval=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='./swav_800ep_pretrain.pth.tar'))))
