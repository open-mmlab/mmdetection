# model settings
model = dict(
    type='CenterNet',
    pretrained='modelzoo://centernet_hg',
    backbone=dict(
        type='DLA',
        base_name='dla34'),
    rpn_head=dict(
        type='CtdetHead', heads=dict(hm=20, wh=2, reg=2)))
cudnn_benchmark = True

train_cfg = dict(a=10)

_valid_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
]
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

test_cfg = dict(
    num_classes=20,
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
            img_scale=(384,384),
            img_norm_cfg=img_norm_cfg,
            _data_rng = np.random.RandomState(123),
            _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32),
            _eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
                                  [-0.5832747, 0.00994535, -0.81221408],
                                  [-0.56089297, 0.71832671, 0.41158938]],
                                 dtype=np.float32),
            max_objs = 50,
            num_classes = 20)
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
                keep_ratio=False,
                input_res=(384, 384),
                img_norm_cfg=img_norm_cfg)
        ])
]

dataset_type = 'VOCDataset'
data_root = 'data/voc/'
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=[
            data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            data_root + 'VOC2012/ImageSets/Main/trainval.txt'
        ],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='Adam', lr=1.25e-4)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = {}
# learning policy
lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=1.0 / 3,
    step=[45, 60])
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
total_epochs = 70
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'data/work_dirs/centernet_dla_pascal'
load_from = None
resume_from = None
workflow = [('train', 1)]
