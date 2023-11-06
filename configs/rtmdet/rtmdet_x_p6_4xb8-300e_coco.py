_base_ = './rtmdet_x_8xb32-300e_coco.py'

model = dict(
    backbone=dict(arch='P6', out_indices=(2, 3, 4, 5)),
    neck=dict(in_channels=[320, 640, 960, 1280]),
    bbox_head=dict(
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32, 64])))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(1280, 1280), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(2560, 2560),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1280, 1280)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1280, 1280), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(1280, 1280),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1280, 1280)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1280, 1280), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    dict(type='Pad', size=(1280, 1280), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8, num_workers=20, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=5, num_workers=20, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epochs = 300
stage2_num_epochs = 20

base_lr = 0.004 * 32 / 256
optim_wrapper = dict(optimizer=dict(lr=base_lr))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

img_scales = [(1280, 1280), (640, 640), (1920, 1920)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=s, keep_ratio=True)
                for s in img_scales
            ],
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [
                dict(
                    type='Pad',
                    size=(1920, 1920),
                    pad_val=dict(img=(114, 114, 114))),
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]
