# dataset settings
dataset_type = 'ADE20KPanopticDataset'
data_root = 'data/ADEChallengeData2016/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    # TODO: the performance of `FixScaleResize` need to check.
    dict(type='FixScaleResize', scale=(2560, 640), backend_args=backend_args),
    dict(type='RandomCrop', crop_size=(640, 640), crop_type='absolute'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ade20k_panoptic_train.json',
        data_prefix=dict(img='images/training/', seg='ade20k_panoptic_train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ade20k_panoptic_val.json',
        data_prefix=dict(img='images/validation/', seg='ade20k_panoptic_val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoPanopticMetric',
    ann_file=data_root + 'ade20k_panoptic_val.json',
    seg_prefix=data_root + 'ade20k_panoptic_val/',
    backend_args=backend_args)
test_evaluator = val_evaluator
