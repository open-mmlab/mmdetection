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
            upsample_cfg=dict(mode='bilinear', align_corners=True)),
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
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=2.5198420997897464,
            scales_per_octave=3,
            ratios=[1.3],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.1, 0.1]),
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
        #alpha=0.4,
        #height_th=9,
        #nms_pre=-1,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=-1))
dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
'''
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='WIDERFaceDataset',
            ann_file='data/WIDERFace/train.txt',
            img_prefix='data/WIDERFace/WIDER_train/',
            min_size=17,
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='RandomSquareCrop',
                    crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[1, 1, 1],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'])
            ])),
    val=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/val.txt',
        img_prefix='data/WIDERFace/WIDER_val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(300, 300),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/val.txt',
        img_prefix='data/WIDERFace/WIDER_val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1100, 1650),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32, pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        min_size=1,
        offset=0))
'''
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/train.txt',
        img_prefix='data/WIDERFace/WIDER_train/',
        min_size=1,
        offset=0,
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='MinIoURandomCrop',
                min_ious=[0.3, 0.45, 0.6, 0.8, 1.0]),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[1, 1, 1],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore'])
        ]),
    val=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/val.txt',
        img_prefix='data/WIDERFace/WIDER_val/',
        min_size=1,
        offset=0,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1100, 1650),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32, pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='WIDERFaceDataset',
        ann_file='data/WIDERFace/val.txt',
        img_prefix='data/WIDERFace/WIDER_val/',
        min_size=1,
        offset=0,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1100, 1650),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32, pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
checkpoint_config = dict(interval=1, type='CheckpointHook')
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(type='OptimizerHook')
lr_config = dict(
    policy='CosineRestart',
    periods=[
        30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30, 30
    ],
    restart_weights=[
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=0.01)
runner = dict(type='EpochBasedRunner', max_epochs=631)
custom_hooks = [dict(type='NumClassCheckHook')]
evaluation = dict(interval=1)
work_dir = './work_dirs/tinaface_r50_fpn_1x_widerface'
gpu_ids = range(0, 1)
