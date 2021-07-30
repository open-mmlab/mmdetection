_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

# model settings
init_cfg = dict(type='Kaiming',
                layer='Conv2d',
                a=2.23606797749979,  # sqrt(5)
                distribution='uniform',
                mode='fan_in',
                nonlinearity='leaky_relu')
model = dict(
    type='YOLOX',
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5, init_cfg=init_cfg),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        csp_num_blocks=1,
        init_cfg=init_cfg),
    bbox_head=dict(
        type='YOLOXHead', num_classes=80, in_channels=128, feat_channels=128, init_cfg=init_cfg),
    # test
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        conf_thr=0.01,  # TODO test 0.001 val 0.01
        score_thr=0.01,  # TODO test 0.001 val 0.01
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=1000))

# dataset settings
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    dict(type='YoloXColorJit'),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad2square=True, pad_val=114.0),
    dict(type='FilterSmallBBox'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('img_norm_cfg',))
]

img_scale = (640, 640)  # h,w

# enable_mixup=True, scale=(0.1, 2)
train_dataset = dict(
    type='MosaicMixUpDataset',
    dataset=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    mosaic_pipeline=[dict(type="RandomAffineOrPerspective", scale=(0.1, 2), border=(-img_scale[0]//2, -img_scale[1]//2))],
    enable_mosaic=True,
    enable_mixup=True,
    pipeline=train_pipeline,
    img_scale=img_scale,
    mixup_scale=(0.8, 1.6))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=img_scale, pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
batch_size = 16  # single gpu
basic_lr_per_img = 0.01 / 64.0

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=train_dataset,
    test=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    val=dict(
        type='CocoDataset',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0,  # We don't need this parameter
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealingWithStop',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=batch_size * basic_lr_per_img,
    warmup_iters=5,  # 5 epoch
    no_aug_epoch=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)

resume_from = None

interval = 10
# ratio_range=(14, 26)
custom_hooks = [
    dict(
        type='YOLOXProcessHook',
        ratio_range=(14, 26),
        img_scale=img_scale,
        no_aug_epoch=15,
        sync_interval=interval,
        priority=48),
    dict(type='EMAHook', priority=49, resume_from=resume_from)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(interval=interval, metric='bbox')
log_config = dict(interval=50)
