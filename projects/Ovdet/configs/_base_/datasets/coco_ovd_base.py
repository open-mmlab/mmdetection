# dataset settings
_base_ = 'mmdet::_base_/datasets/coco_detection.py'

data_root = 'data/coco/'

train_dataloader = dict(
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        ann_file='wusize/instances_train2017_base.json',
        data_prefix=dict(img='train2017/'),
    ))
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
