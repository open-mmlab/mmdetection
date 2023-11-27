# dataset settings
_base_ = 'mmdet::_base_/datasets/coco_detection.py'
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
file_client_args = dict(backend='disk')
branch_field = ['det_batch', 'caption_batch']
det_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
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
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PackDetInputs')
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        caption_batch=dict(
            type='PackDetInputs',
            meta_keys=[
                'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                'flip', 'flip_direction', 'captions', 'tags', 'image_ids'
            ]))
]
det_dataset = dict(
    type='CocoDataset',
    data_root=data_root,
    ann_file='wusize/instances_train2017_base.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=det_pipeline)

ovd_dataset = dict(
    type='CocoCaptionOVDDataset',
    data_root=data_root,
    ann_file='wusize/captions_train2017_tags_allcaps.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=ovd_pipeline)
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
