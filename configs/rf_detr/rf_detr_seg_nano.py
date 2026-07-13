# Copyright (c) OpenMMLab. All rights reserved.
_base_ = './rf_detr_seg_preview.py'

image_size = 312

model = dict(
    model_config_type='seg_nano',
    model_config=dict(
        resolution=image_size,
        positional_encoding_size=image_size // 12,
        num_queries=100,
        num_select=100,
        pretrain_weights=None),
    train_config=dict(output_dir='work_dirs/rf_detr_seg_nano'))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
