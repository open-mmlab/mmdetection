_base_ = 'fcos_r50_center_fpn_gn-head_1x_coco.py'
# model settings
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    )
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type="RandomCrop", crop_size=(800, 800), crop_type='absolute'),
    dict(type='Resize', img_scale=(800, 800), ratio_range=(0.5, 1.4), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
#dict(type='Rotate', level=1, max_rotate_angle=30, prob=0.6),
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800),(1400, 1024),(2048, 2048),(1024, 512)],
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
#fp16 = dict(loss_scale='dynamic')  
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
optimizer = dict(
    lr=0.0025, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))  #学习率的设置尤为关键：lr = 0.00125*batch_size
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
load_from = '/mnt/d/Github/mmdetection/project/LogDetMini/workdir/fcos/r50dcn/epoch_12.pth'