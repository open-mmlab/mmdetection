_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../_base_/datasets/dsdl.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=365)))

# dsdl dataset settings

# please visit our platform [OpenDataLab](https://opendatalab.com/)
# to downloaded dsdl dataset.
data_root = 'data/Objects365'
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
        test_mode=True,
    ))
test_dataloader = val_dataloader

default_hooks = dict(logger=dict(type='LoggerHook', interval=1000), )
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[1, 2],
        gamma=0.1)
]
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
