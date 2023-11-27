# dataset settings
_base_ = 'mmdet::_base_/datasets/lvis_v1_instance.py'

data_root = 'data/coco/'

train_dataloader = dict(
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(dataset=dict(ann_file='wusize/lvis_v1_train.json')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='wusize/lvis_v1_val.json',
        data_prefix=dict(img='')))

val_evaluator = dict(ann_file=data_root + 'wusize/lvis_v1_val.json')
