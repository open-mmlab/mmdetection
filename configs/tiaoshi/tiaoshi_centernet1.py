dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(
                type='LoadImageFromFile', to_float32=True, color_type='color'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='RandomCenterCropPad',
                crop_size=(512, 512),
                ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_pad_mode=None),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomCenterCropPad',
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=['logical_or', 31],
                        test_pad_add_pix=1),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg', 'border'),
                        keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomCenterCropPad',
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=['logical_or', 31],
                        test_pad_add_pix=1),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg', 'border'),
                        keys=['img'])
                ])
        ]))
evaluation = dict(interval=5, metric='bbox')
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[90, 120])
runner = dict(type='EpochBasedRunner', max_epochs=140)
checkpoint_config = dict(interval=5)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='CenterNet',
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet', depth=18, norm_eval=False, norm_cfg=dict(type='BN')),
    neck=dict(
        type='CTResNetNeck',
        in_channel=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=80,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))
work_dir = './work_dirs/tiaoshi_centernet'
gpu_ids = range(0, 1)
