# dataset settings
dataset_type = 'ReIDDataset'
data_root = 'data/MOT17/'

backend_args = None
# data pipeline
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(
                type='LoadImageFromFile',
                backend_args=backend_args,
                to_float32=True),
            dict(
                type='Resize',
                scale=(128, 256),
                keep_ratio=False,
                clip_object_border=False),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        ]),
    dict(type='PackReIDInputs', meta_keys=('flip', 'flip_direction'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
    dict(type='Resize', scale=(128, 256), keep_ratio=False),
    dict(type='PackReIDInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        triplet_sampler=dict(num_ids=8, ins_per_id=4),
        data_prefix=dict(img_path='reid/imgs'),
        ann_file='reid/meta/train_80.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        triplet_sampler=None,
        data_prefix=dict(img_path='reid/imgs'),
        ann_file='reid/meta/val_20.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(type='ReIDMetrics', metric=['mAP', 'CMC'])
test_evaluator = val_evaluator
