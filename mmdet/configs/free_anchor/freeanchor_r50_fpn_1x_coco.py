if '_base_':
    from ..retinanet.retinanet_r50_fpn_1x_coco import *
from mmdet.models.dense_heads.free_anchor_retina_head import FreeAnchorRetinaHead
from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss

model.merge(
    dict(
        bbox_head=dict(
            _delete_=True,
            type=FreeAnchorRetinaHead,
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type=AnchorGenerator,
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type=DeltaXYWHBBoxCoder,
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_bbox=dict(type=SmoothL1Loss, beta=0.11, loss_weight=0.75))))

optim_wrapper.merge(dict(clip_grad=dict(max_norm=35, norm_type=2)))
