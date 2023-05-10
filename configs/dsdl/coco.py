_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../_base_/datasets/dsdl.py'
]

# dsdl dataset settings

# please visit our platform [OpenDataLab](https://opendatalab.com/)
# to downloaded dsdl dataset.
data_root = 'data/COCO2017'
img_prefix = 'original'
train_ann = 'dsdl/set-train/train.yaml'
val_ann = 'dsdl/set-val/val.yaml'
specific_key_path = dict(ignore_flag='./annotations/*/iscrowd')

train_dataloader = dict(
    dataset=dict(
        specific_key_path=specific_key_path,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img_path=img_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    ))

val_dataloader = dict(
    dataset=dict(
        specific_key_path=specific_key_path,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img_path=img_prefix),
    ))
test_dataloader = val_dataloader
