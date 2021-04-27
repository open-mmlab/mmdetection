_base_ = [
    '../_base_/datasets/wider_face.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)

model = dict(
    type='TinaFace',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        norm_eval=False,
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        style='pytorch'),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=6,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            upsample_cfg=dict(mode='bilinear')),
        dict(
            type='Inception',
            in_channel=256,
            num_levels=6,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            share=True)
    ],
    bbox_head=dict(
        type='TinaFaceHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=2 ** (4 / 3),
            scales_per_octave=3,
            ratios=[1.3],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='DIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.35,
            neg_iou_thr=0.35,
            min_pos_iou=0.35,
            ignore_iof_thr=-1,
            gpu_assign_thr=100),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        alpha=0.4,
        height_th=9,
        nms_pre=-1,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=-1),
)

optimizer = dict(type='SGD', lr=3.75e-3, momentum=0.9, weight_decay=5e-4)



test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1100, 1650),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(test=dict(min_size=1, offset=0, pipeline=test_pipeline))
