# dataset settings
dataset_type = 'OpenImagesDataset'
data_root = 'data/OpenImages/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, denorm_bbox=True),
    dict(type='Resize', img_scale=(1024, 800), keep_ratio=True),
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
        img_scale=(1024, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,  # workers_per_gpu > 0 may occur out of memory
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/oidv6-train-annotations-bbox.csv',
        img_prefix=data_root + 'OpenImages/train/',
        label_file=data_root + 'annotations/class-descriptions-boxable.csv',
        hierarchy_file=data_root +
        'annotations/bbox_labels_600_hierarchy.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/validation-annotations-bbox.csv',
        img_prefix=data_root + 'OpenImages/validation/',
        label_file=data_root + 'annotations/class-descriptions-boxable.csv',
        hierarchy_file=data_root +
        'annotations/bbox_labels_600_hierarchy.json',
        meta_file=data_root + 'annotations/validation-image-metas.pkl',
        image_level_ann_file=data_root +
        'annotations/validation-annotations-human-imagelabels-boxable.csv',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/validation-annotations-bbox.csv',
        img_prefix=data_root + 'OpenImages/validation/',
        label_file=data_root + 'annotations/class-descriptions-boxable.csv',
        hierarchy_file=data_root +
        'annotations/bbox_labels_600_hierarchy.json',
        meta_file=data_root + 'annotations/validation-image-metas.pkl',
        image_level_ann_file=data_root +
        'annotations/validation-annotations-human-imagelabels-boxable.csv',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
