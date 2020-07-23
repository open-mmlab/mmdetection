dataset_type = 'WflwDataset'
data_root = '/mnt/data-home/sebastien/data/public/WFLW/'

img_scale = (384, 384)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='FocusOnFace', out_size=384),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoint=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='Corrupt', corruption='gaussian_noise'),
    dict(type='Corrupt', corruption='jpeg_compression'),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='KeypointsToHeatmaps', sigma=3),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'heatmaps']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FocusOnFace', out_size=384),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=img_scale, keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=20,
    train=dict(
        type=dataset_type,
        ann_file=data_root +
        'WFLW_annotations/list_98pt_rect_attr_train_test/' +
        'list_98pt_rect_attr_train.txt',
        img_prefix=data_root + 'WFLW_images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'WFLW_annotations/list_98pt_rect_attr_train_test/' +
        'list_98pt_rect_attr_test.txt',
        img_prefix=data_root + 'WFLW_images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'WFLW_annotations/list_98pt_rect_attr_train_test/' +
        'list_98pt_rect_attr_test.txt',
        img_prefix=data_root + 'WFLW_images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='keypoint', kpts_thr=0.1)
