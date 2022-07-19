# dataset settings
_base_ = 'lvis_v0.5_instance.py'
dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/lvis_v1_train.json',
            data_prefix=dict(img=''))))
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v1_val.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/lvis_v1_val.json')
test_evaluator = val_evaluator
