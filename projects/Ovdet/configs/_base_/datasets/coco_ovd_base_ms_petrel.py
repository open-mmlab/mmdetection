# dataset settings
_base_ = 'mmdet::_base_/datasets/coco_detection.py'

data_root = 'data/coco/'
# file_client_args = dict(backend='disk')

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/': 'openmmlab:s3://openmmlab/datasets/detection/',
        'data/': 'openmmlab:s3://openmmlab/datasets/detection/'
    }))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        ann_file='wusize/instances_train2017_base.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline))
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wusize/instances_val2017_base.json',
        metric='bbox',
        prefix='Base',
        format_only=False),
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wusize/instances_val2017_novel.json',
        metric='bbox',
        prefix='Novel',
        format_only=False)
]
test_evaluator = val_evaluator
