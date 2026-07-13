# Copyright (c) OpenMMLab. All rights reserved.
default_scope = 'mmdet'

custom_imports = dict(
    imports=['mmdet.models.detectors.rf_detr'], allow_failed_imports=False)

data_root = 'data/coco/'
train_ann_file = 'annotations/instances_train2017.json'
val_ann_file = 'annotations/instances_val2017.json'
test_ann_file = val_ann_file
train_data_prefix = 'train2017/'
val_data_prefix = 'val2017/'
test_data_prefix = val_data_prefix

num_classes = 90
metainfo = dict(classes=tuple(str(i) for i in range(num_classes)))
load_from = None

image_size = 512

model = dict(
    type='RFDETR',
    num_classes=num_classes,
    model_config_type='small',
    train_config_type='detection',
    score_thr=0.0,
    model_config=dict(
        num_classes=num_classes,
        resolution=image_size,
        positional_encoding_size=image_size // 16,
        num_queries=300,
        num_select=300,
        group_detr=13,
        dec_layers=3,
        projector_scale=['P4'],
        pretrain_weights=None,
        device='cpu',
        amp=True),
    train_config=dict(
        dataset_dir=data_root,
        output_dir='work_dirs/rf_detr_small',
        group_detr=13,
        cls_loss_coef=1.0,
        batch_size=2,
        grad_accum_steps=1,
        use_ema=True),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

test_dataloader = val_dataloader.copy()
test_dataloader['dataset'] = test_dataloader['dataset'].copy()
test_dataloader['dataset']['ann_file'] = test_ann_file
test_dataloader['dataset']['data_prefix'] = dict(img=test_data_prefix)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator.copy()
test_evaluator['ann_file'] = data_root + test_ann_file

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', by_epoch=True, milestones=[11], gamma=0.1)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_last=True,
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
resume = False
auto_scale_lr = dict(enable=False, base_batch_size=16)
