# dataset settings
dataset_type = 'ADE20KPanopticDataset'
data_root = 'data/ADEChallengeData2016/'

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
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
