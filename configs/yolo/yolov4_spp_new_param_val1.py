_base_ = '../_base_/default_runtime.py'
# model settings
model = dict(
    type='YOLOV4',
    # TODO: fix pretrained model
    # pretrained='../checkpoints/yolov4/yolov4.conv.137.pth',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        csp_on=True,
        norm_cfg=dict(type='BN', requires_grad=True, eps=1e-04,
                      momentum=0.03)),
    neck=dict(
        type='YOLOV4Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128],
        spp_on=True,
        spp_pooler_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', requires_grad=True, eps=1e-04, momentum=0.03),
    ),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=80,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(142, 110), (192, 243), (459, 401)],
                        [(36, 75), (76, 55), (72, 146)],
                        [(12, 16), (19, 36), (40, 28)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        norm_cfg=dict(type='BN', requires_grad=True, eps=1e-04, momentum=0.03),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=37.4,
            reduction='mean'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=64.3,
            reduction='mean'),
        loss_bbox=dict(type='GIoULoss', loss_weight=3.54, reduction='mean'),
    ))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='GridAssigner', pos_iou_thr=1.0, neg_iou_thr=0.2, min_pos_iou=0))
test_cfg = dict(
    nms_pre=4000,
    min_bbox_size=2,
    score_thr=0.001,
    conf_thr=0.001,
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 1)),
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    #     min_crop_size=0.3),
    dict(type='Resize', img_scale=[(448, 448), (448, 448)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val1.json',
        # ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val1.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val1.json',
        img_prefix=data_root + 'val2017/',
        # if the test weight is transformed from ultralytic/yolov3
        # ann_file=data_root + 'annotations/coco_yolo_5k.json',
        # img_prefix=data_root + 'val2014/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.937, weight_decay=0.000484)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])
# runtime settings
total_epochs = 273
evaluation = dict(interval=10, metric=['bbox'])

log_config = dict(  # config to register logger hook
    interval=1,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook')
    ])  # The logger used to record the training process.
