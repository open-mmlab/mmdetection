# Copyright (c) 2019 Western Digital Corporation or its affiliates.

# model settings
model = dict(
    type='YoloNet',
    pretrained='checkpoints/darknet_state_dict_only.pth',
    backbone=dict(
        type='DarkNet53',),
    neck=dict(
        type='YoloNeck',),
    bbox_head=dict(
        type='YoloHead',))
# training and testing settings
train_cfg = dict(
    # assigner=dict(
    #     type='MaxIoUAssigner',
    #     pos_iou_thr=0.5,
    #     neg_iou_thr=0.4,
    #     min_pos_iou=0,
    #     ignore_iof_thr=-1),
    # allowed_border=-1,
    # pos_weight=-1,
    one_hot_smoother=0.,
    ignore_config=0.5,
    xy_use_logit=False,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_thr=0.45),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(type='Expand',
         mean=img_norm_cfg['mean'],
         to_rgb=img_norm_cfg['to_rgb'],
         ratio_range=(1, 2)
         ),
    dict(type='MinIoURandomCrop',
         min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
         min_crop_size=0.3
         ),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
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
        img_scale=(608, 608),
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
    imgs_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline,
        # img_scale=[(320, 320), (608, 608)],
        # multiscale_mode='range',
        # img_norm_cfg=img_norm_cfg,
        # size_divisor=32,
        # flip_ratio=0.5,
        # with_mask=False,
        # with_crowd=False,
        # with_label=True,
        # extra_aug=dict(
        #     photo_metric_distortion=dict(
        #         brightness_delta=32,
        #         contrast_range=(0.5, 1.5),
        #         saturation_range=(0.5, 1.5),
        #         hue_delta=18),
        #     expand=dict(
        #         mean=img_norm_cfg['mean'],
        #         to_rgb=img_norm_cfg['to_rgb'],
        #         ratio_range=(1, 2)),
        #     random_crop=dict(
        #         min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), min_crop_size=0.3)),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        # img_scale=(608, 608),
        # img_norm_cfg=img_norm_cfg,
        # size_divisor=32,
        # flip_ratio=0,
        # with_mask=False,
        # with_crowd=False,
        # with_label=True
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        # img_scale=(608, 608),
        # img_norm_cfg=img_norm_cfg,
        # size_divisor=32,
        # flip_ratio=0,
        # with_mask=False,
        # with_crowd=False,
        # with_label=False,
        # test_mode=True
    )
)
# optimizer
optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 273
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/retinanet_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
