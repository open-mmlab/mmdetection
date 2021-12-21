_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)
model = dict(
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        stem_channels=40,
        norm_cfg=norm_cfg,
        scale=3,
        with_cp=True,
        dropout=0.3,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/mnt/lustre/jiangyitong1/mmdetection/checkpoints/converted_b3_2.pyth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[48, 136, 384],
        start_level=0,
        out_channels=160,
        num_outs=5),
    test_cfg=dict(
        nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian')),
    bbox_head=dict(
        type='RetinaSepBNHead',
        num_ins=5,
        in_channels=160,
        loss_cls=dict(gamma=1.5, ),
        norm_cfg=norm_cfg),
)
# lr = 0.08 * (batch_size / 64)
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=4e-5)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)
total_epochs = 300
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=10)
img_norm_cfg = dict(
    # The mean and std is used in PyCls
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(896, 896),
        ratio_range=(0.1, 2.0),
        keep_ratio=True,
    ),
    dict(
        type='RandomCrop',
        crop_size=(896, 896),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(896, 896),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=3,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
