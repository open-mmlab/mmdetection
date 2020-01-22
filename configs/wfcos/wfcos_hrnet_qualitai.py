# model settings
model = dict(
    type='WFCOS',
    pretrained='open-mmlab://msra/hrnetv2_w32',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    neck=dict(
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256,
        stride=2,
        num_outs=5),
    bbox_head=dict(
        type='WFCOSHead',
        num_classes=2,
        in_channels=256,
        max_energy=20,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=.1),
        loss_bbox=dict(
            type='IoULoss',
            loss_weight=1.0),
        loss_energy=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=5.0,
            loss_weight=4.0,
            reduction='sum'
        ),
        split_convs=False,
        r=250.
    ))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.3,
    nms=dict(type='nms', iou_thr=0.01),
    max_per_img=1000)
# dataset settings
dataset_type = 'QualitaiDataset'
data_root = 'data/qualitai/'
img_norm_cfg = dict(
    mean=[32.20495642019232, 31.513648345703196, 36.627047367261675],
    std=[34.395634168647526, 36.89673991173119, 38.85190978611362],
    to_rgb=False)
img_scale_train = (1024, 800)
img_scale_test = (1024, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale_train, keep_ratio=True),
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
        img_scale=img_scale_test,
        flip=False,
        transforms=[
            # dict(type='RandomCrop', crop_size=(640, 800)),
            dict(type='Resize', img_scale=img_scale_test, keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'qualitai_training_bad.json',
        img_prefix=data_root + 'train/bad/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'qualitai_validation_bad.json',
        img_prefix=data_root + 'valid/bad/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'qualitai_validation_bad.json',
        img_prefix=data_root + 'valid/bad/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0008,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    step=[16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 40
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/qualitai'
load_from = None
# load_from = work_dir + '/epoch_4.pth'
resume_from = None
# resume_from = work_dir + '/epoch_4.pth'
workflow = [('train', 1)]

# wandb settings
wandb_cfg = dict(
    entity='warp-net',
    project='qualitai',
    dryrun=False
)
