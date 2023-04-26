_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    # '../_base_/datasets/dsdl.py'
]

# model setting
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

# dsdl dataset settings

# please visit our platform [OpenDataLab](https://opendatalab.com/)
# to downloaded dsdl dataset.
dataset_type = 'DSDLDetDataset'
data_root_07 = 'data/VOC07-det'
data_root_12 = 'data/VOC12-det'
img_prefix = 'original'

train_ann = 'dsdl/set-train/train.yaml'
val_ann = 'dsdl/set-val/val.yaml'
test_ann = 'dsdl/set-test/test.yaml'

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]

specific_key_path = dict(ignore_flag='./objects/*/difficult', )

train_dataloader = dict(
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type=dataset_type,
                    specific_key_path=specific_key_path,
                    data_root=data_root_07,
                    ann_file=train_ann,
                    data_prefix=dict(img_path=img_prefix),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline),
                dict(
                    type=dataset_type,
                    specific_key_path=specific_key_path,
                    data_root=data_root_07,
                    ann_file=val_ann,
                    data_prefix=dict(img_path=img_prefix),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline),
                dict(
                    type=dataset_type,
                    specific_key_path=specific_key_path,
                    data_root=data_root_12,
                    ann_file=train_ann,
                    data_prefix=dict(img_path=img_prefix),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline),
                dict(
                    type=dataset_type,
                    specific_key_path=specific_key_path,
                    data_root=data_root_12,
                    ann_file=val_ann,
                    data_prefix=dict(img_path=img_prefix),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline),
            ])))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        specific_key_path=specific_key_path,
        data_root=data_root_07,
        ann_file=test_ann,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', metric='bbox')
# val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

# training schedule, voc dataset is repeated 3 times, in
# `_base_/datasets/voc0712.py`, so the actual epoch = 4 * 3 = 12
max_epochs = 4
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
