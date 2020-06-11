# dataset settings
dataset_type = 'DeepFashionDataset'
data_root = 'data/DeepFashion/In-shop/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(750, 1101), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(750, 1101),
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
    imgs_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/DeepFashion_segmentation_query.json',
        img_prefix=data_root + 'Img/',
        pipeline=train_pipeline,
        data_root=data_root),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/DeepFashion_segmentation_query.json',
        img_prefix=data_root + 'Img/',
        pipeline=test_pipeline,
        data_root=data_root),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/DeepFashion_segmentation_gallery.json',
        img_prefix=data_root + 'Img/',
        pipeline=test_pipeline,
        data_root=data_root))
evaluation = dict(interval=5, metric=['bbox', 'segm'])
