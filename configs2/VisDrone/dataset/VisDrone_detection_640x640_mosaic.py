dataset_type = 'VisDroneCocoDataset'
# data_root = '../data/dense/'
data_root = '../data/VisDrone2019/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (640, 640)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=45,
        interpolation=1,
        p=0.5),  # 0.5
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[-0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.4),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),  # 0.1
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),  # 0.2
    dict(type='ChannelShuffle', p=0.1),  # 0.1
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type='MotionBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),  # 0.1
]

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='Resize', img_scale=[(640, 640), (960, 960)], keep_ratio=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1280, 1280), (1920, 1920)],  # (1280, 1024), (1920, 1536)
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
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
