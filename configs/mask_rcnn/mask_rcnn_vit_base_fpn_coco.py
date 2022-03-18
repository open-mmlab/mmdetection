_base_ = [
    '../_base_/models/mask_rcnn_vit_base_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1024, 1024),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(1024, 1024)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(1024, 1024)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'bias': dict(decay_mult=0.)
        }))

# mixed precision
fp16 = dict(loss_scale='dynamic')

# learning rate schedule
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_iters=25,
    warmup_by_epoch=True,
    by_epoch=False)

# training epochs
runner = dict(type='EpochBasedRunner', max_epochs=100)
