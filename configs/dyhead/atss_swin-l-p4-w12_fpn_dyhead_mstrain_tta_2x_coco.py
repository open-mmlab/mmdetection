_base_ = './atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco.py'

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(3000, 400), (3000, 500), (3000, 600), (3000, 700),
                   (3000, 800), (3000, 900), (3000, 1000), (3000, 1100),
                   (3000, 1200), (3000, 1300), (3000, 1400), (3000, 1800)],
        obj_scale_range=[(96, 10000), (96, 10000), (64, 10000), (64, 10000),
                         (0, 10000), (0, 10000), (0, 10000), (0, 256),
                         (0, 256), (0, 192), (0, 192), (0, 96)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True, backend='pillow'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'obj_range')),
        ])
]

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
