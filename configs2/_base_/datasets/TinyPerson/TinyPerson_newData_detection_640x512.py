dataset_type = 'CocoDataset'
# data_root = '../data/dense/'
data_root = '../data/tiny_set/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    dict(type='LoadSubImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(640, 512), (960, 768)], keep_ratio=True),
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
    dict(type='LoadSubImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1280, 1024), (1920, 1536)],  # (1280, 1024), (1920, 1536)
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
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotation/Sub_training/sub_train/overlap30/train_overlap30.json',
        ann_file=data_root + 'divide_640x512_overlap30_annotations/train/train_overlap30.json',
        img_prefix=data_root + 'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'divide_640x512_overlap30_annotations/test/tiny_set_test_single.json',
        img_prefix=data_root + 'test',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'divide_640x512_overlap30_annotations/test/tiny_set_test_single.json',
        img_prefix=data_root + 'test',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox',save_best='bbox_mAP')
