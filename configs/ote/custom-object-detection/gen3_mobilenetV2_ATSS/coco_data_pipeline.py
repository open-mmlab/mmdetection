dataset_type = 'CocoDataset'
img_size = (992, 736)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672), (992, 800)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        adaptive_repeat_times=True,
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file='data/coco/annotations/instances_train2017.json',
            img_prefix='data/coco/train2017',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017',
        test_mode=True,
        pipeline=test_pipeline))
