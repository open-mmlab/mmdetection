# model settings
input_size = 256
image_width, image_height = input_size, input_size
width_mult = 1.0
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True
    ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        num_classes=1,
        in_channels=(int(width_mult * 96), int(width_mult * 320)),
        anchor_generator=dict(
            type='SSDAnchorGeneratorClustered',
            strides=(16, 32),
            widths=[
                [image_width * x for x in
                 [0.015411783166343854, 0.033018232306549156, 0.04467156688464953,
                  0.0610697815328886]],
                [image_width * x for x in
                 [0.0789599954420517, 0.10113984043326349, 0.12805187473050397, 0.16198319380154758,
                  0.21636496806213493]],

            ],
            heights=[
                [image_height * x for x in
                 [0.05032631418898226, 0.10070800135152037, 0.15806180366055939,
                  0.22343401646383804]],
                [image_height * x for x in
                 [0.300881401352503, 0.393181898580379, 0.4998807213337051, 0.6386035764261081,
                  0.8363451552091518]],

            ],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2), ),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=True),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        use_giou=False,
        use_focal=False,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True
# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../data/airport/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.1),
    dict(type='Resize', img_scale=(input_size, input_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_size, input_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=122,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            classes=('person',),
            ann_file=data_root + 'annotation_person_train.json',
            min_size=20,
            img_prefix=data_root + 'train',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        classes=('person',),
        ann_file=data_root + 'annotation_person_val.json',
        img_prefix=data_root + 'val',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('person',),
        ann_file=data_root + 'annotation_person_val.json',
        img_prefix=data_root + 'val',
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 3,
    step=[8, 15, 18])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/person-detection-0200'
load_from = None
resume_from = None
workflow = [('train', 1)]
