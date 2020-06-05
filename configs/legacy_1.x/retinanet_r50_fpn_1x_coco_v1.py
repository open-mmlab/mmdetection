_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(
        type='RetinaHead',
        anchor_generator=dict(
            type='LegacyAnchorGenerator',
            center_offset=0.5,
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(type='LegacyDeltaXYWHBBoxCoder'),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
