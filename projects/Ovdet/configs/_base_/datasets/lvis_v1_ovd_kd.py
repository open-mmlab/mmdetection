# dataset settings
_base_ = 'mmdet::_base_/datasets/lvis_v1_instance.py'
dataset_type = 'LVISV1Dataset'

data_root = 'data/coco/'

file_client_args = dict(backend='disk')

branch_field = ['det_batch', 'kd_batch']
det_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PackDetInputs')
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        det_batch=dict(type='PackDetInputs'))
]
ovd_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PackDetInputs')
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        kd_batch=dict(type='PackDetInputs'))
]

det_dataset = dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='wusize/lvis_v1_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=det_pipeline))

ovd_dataset = dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='wusize/lvis_v1_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=ovd_pipeline))

batch_split = [1, 1]
train_dataloader = dict(
    batch_size=sum(batch_split),
    num_workers=sum(batch_split),
    persistent_workers=True,
    sampler=dict(
        type='CustomGroupMultiSourceSampler',
        batch_size=sum(batch_split),
        source_ratio=batch_split),
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[det_dataset, ovd_dataset]))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='wusize/lvis_v1_val.json',
        data_prefix=dict(img='')))

val_evaluator = dict(ann_file=data_root + 'wusize/lvis_v1_val.json')
