_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
image_size = (640, 640)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

model = dict(
    data_preprocessor=dict(pad_size_divisor=64, batch_augments=batch_augments),
    backbone=dict(norm_cfg=norm_cfg, norm_eval=False),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(bbox_head=dict(norm_cfg=norm_cfg)))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8, num_workers=4, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# learning policy
max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=2)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30, 40],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True),
    clip_grad=None)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
