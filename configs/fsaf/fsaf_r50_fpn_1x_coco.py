_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FSAF',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='FSAFHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=1,
        scales_per_octave=1,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_normalizer=1.0,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        loss_bbox=dict(type='IoULossTBLR',
                       eps=1e-6,
                       loss_weight=1.0,
                       reduction='none')))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='EffectiveAreaAssigner',
        pos_area_thr=0.2,
        neg_area_thr=0.2,
        min_pos_iof=0.01),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))