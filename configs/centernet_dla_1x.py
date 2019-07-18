# model settings
model = dict(
    type='CenterNet',
    pretrained='modelzoo://centernet_hg',
    backbone=dict(
        type='DLA',
        base_name='dla34'),
    rpn_head=dict(
        type='CtdetHead', heads=dict(hm=80, wh=2, reg=2)))
cudnn_benchmark = True

train_cfg = dict(a=10)

_valid_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]
img_norm_cfg = dict(
    mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=True)

test_cfg = dict(
    num_classes=80,
    valid_ids={i + 1: v
               for i, v in enumerate(_valid_ids)},
    img_norm_cfg=img_norm_cfg,
    debug=0)

import numpy as np
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CtdetTrainTransforms',
            flip_ratio=0.5,
            size_divisor=31,
            keep_ratio=False,
            img_scale=(512,512),
            img_norm_cfg=img_norm_cfg,
            max_objs = 128,
            num_classes = 80,
            _data_rng = np.random.RandomState(123),
            _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                     dtype=np.float32),
            _eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
                                      [-0.5832747, 0.00994535, -0.81221408],
                                      [-0.56089297, 0.71832671, 0.41158938]],
                                     dtype=np.float32)
                                    )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1, 1),
        flip=True,
        transforms=[
            dict(type='CtdetTestTransforms',
                size_divisor=31,
                keep_ratio=True,
                input_res=(512, 512),
                img_norm_cfg=img_norm_cfg)
        ])
]


dataset_type = 'CocoDataset'
data_root = 'data/coco/'
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='Adam', lr=2.5e-4)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = {}
# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=1.0 / 3,
    step=[90, 120])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 140
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'data/work_dirs/centernet_dla_pascal'
load_from = None
resume_from = None
workflow = [('train', 1)]
