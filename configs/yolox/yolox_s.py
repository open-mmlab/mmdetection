_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

data_root = 'data/coco/'

# model settings
model = dict(
    type='YOLOX',
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        csp_num_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=80, in_channels=128, feat_channels=128),
    # test
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        conf_thr=0.01,  # TODO test 0.001 val 0.01
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=1000))

img_norm_cfg = dict(
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    to_rgb=True)

train_pipeline = [
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('img_norm_cfg', ))
]

name = 'train2017/'
annotations = 'annotations/instances_train2017.json'

sub_dataset = dict(
    type='COCODataset',
    ann_file=data_root + annotations,
    img_prefix=data_root + name,
    pipeline=None,
    filter_empty_gt=False,
)

train_dataset = dict(
    type='MosaicDetection',
    dataset=sub_dataset,
    ann_file=data_root + annotations,
    img_prefix=data_root + name,
    pipeline=train_pipeline,
    enable_mixup=True,  # tiny cfg
    scale=(0.1, 2))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(640, 640), pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
batch_size = 8  # single gpu
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
    lr=0,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealingWithNoAugIter',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=batch_size * basic_lr_per_img,
    warmup_iters=5,  # 5 epoch
    no_aug_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=300)

resume_from = None

interval = 10
evaluation = dict(interval=interval, metric='bbox')
custom_hooks = [
    dict(
        type='ProcessHook',
        random_size=(10, 20),
        no_aug_epochs=15,
        eval_interval=interval,
        priority=48),
    dict(type='EMAHook', priority=49, resume_from=resume_from)
]
log_config = dict(interval=50)
checkpoint_config = dict(interval=interval)
