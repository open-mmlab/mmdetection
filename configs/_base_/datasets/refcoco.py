# dataset settings
dataset_type = 'RefCOCODataset'
data_root = 'data/refcoco/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'image_id'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'image_id'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='train2014/'),
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='train',
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
        data_prefix=dict(img='train2014/'),
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='val',
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='train2014/'),
        ann_file='refcoco/instances.json',
        split_file='refcoco/refs(unc).p',
        split='testA',  # or 'testB'
        pipeline=test_pipeline,
        backend_args=backend_args))

# TODO: set the metrics
