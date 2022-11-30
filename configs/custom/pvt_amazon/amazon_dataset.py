# dataset settings
dataset_type = 'AmazonDataset'
data_root = '/mmdetection/data/amazon/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(768, 512), (768, 600)], keep_ratio=True),
        #type='Resize', img_scale=[(1024, 768), (1024, 600)], keep_ratio=True),
    dict(
            type='RandomCrop',
            crop_type='relative_range',
            crop_size=(0.7, 1.0),
            allow_negative_crop=True),
    dict(type='RandomAffine'),
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
        img_scale=(768, 512),
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
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'train/coco_train.json',
            img_prefix=data_root + 'train/image_2/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'val/coco_val.json',
        img_prefix=data_root + 'val/image_2/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'val/coco_val.json',
        img_prefix=data_root + 'val/image_2/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
#log_level="INFO"