dataset_type = 'NOUSDataset'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromNOUSDataset', to_float32=False),
    dict(type='LoadAnnotationFromNOUSDataset', with_bbox=True, with_label=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(2, 2)),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromNOUSDataset', to_float32=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        nous_dataset=None,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        nous_dataset=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        nous_dataset=None,
        pipeline=test_pipeline))
