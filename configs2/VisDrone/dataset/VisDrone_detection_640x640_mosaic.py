dataset_type = 'VisDroneCocoDataset'
# data_root = '../data/dense/'
data_root = '../data/VisDrone2019/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (640, 640)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
    dynamic_scale=img_scale)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=img_scale, pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    # train=dict(
    #     type=dataset_type,
    #     # ann_file=data_root + 'annotation/Sub_training/sub_train/overlap30/train_overlap30.json',
    #     # ann_file=data_root + 'VisDrone2019_slice640_overlap40/train_640_00625.json',
    #     # img_prefix=data_root + 'VisDrone2019_slice640_overlap40/train_images_640_00625',
    #     ann_file=data_root + 'train.json',
    #     img_prefix=data_root + 'VisDrone2019-DET-train/images',
    #     pipeline=train_pipeline),
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train.json',
            img_prefix=data_root + 'VisDrone2019-DET-train/images',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False,
        ),
        pipeline=train_pipeline,
        dynamic_scale=img_scale),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'VisDrone2019_slice640_overlap40/val_640_00625.json',
        # img_prefix=data_root + 'VisDrone2019_slice640_overlap40/val_images_640_00625',

        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'VisDrone2019_slice640_overlap40/test-dev_640_00625.json',
        # img_prefix=data_root + 'VisDrone2019_slice640_overlap40/test-dev_images_640_00625',
        ann_file=data_root + 'test-dev.json',
        img_prefix=data_root + 'VisDrone2019-DET-test-dev/images',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_s')
