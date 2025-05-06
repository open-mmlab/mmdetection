# _base_/nuscenes.py - Base configuration for NuScenes dataset

_base_ = ['./default_runtime.py']

# NuScenes classes
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/nuscenes/'

# Data pipeline for training
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(900, 1600)],
                keep_ratio=True)
        ]]),
    dict(type='PackDetInputs')
]

# Test/validation pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(900, 1600), keep_ratio=True), 
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# DataLoader settings
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/mini_nuscenes_2d_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names))
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/mini_nuscenes_2d_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names))
)

test_dataloader = val_dataloader

# Evaluator config
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/nuscenes_2d_val.json',
    metric=['bbox'],
    classwise=True,  # Track per-class performance
)

test_evaluator = val_evaluator

# Common visualization hooks for NuScenes
custom_hooks = [
    dict(
        type='DetVisualizationHook',
        draw=True,          
        interval=200,       
        show=False,
    ),
]