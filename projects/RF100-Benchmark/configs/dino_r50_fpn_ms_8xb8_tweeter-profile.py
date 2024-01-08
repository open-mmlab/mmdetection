_base_ = '../../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py'

custom_imports = dict(
    imports=['projects.RF100-Benchmark'], allow_failed_imports=False)

data_root = 'rf100/tweeter-profile/'
class_name = ('profile_info', )
num_classes = len(class_name)
metainfo = dict(classes=class_name)
image_scale = (640, 640)

model = dict(
    backbone=dict(
        norm_eval=False, norm_cfg=dict(requires_grad=True), frozen_stages=-1),
    bbox_head=dict(num_classes=int(num_classes)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_scale,
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_scale),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type='RF100CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            ann_file='train/_annotations.coco.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        type='RF100CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        pipeline=test_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='RF100CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

max_epochs = 25
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=200),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[18, 22],
        gamma=0.1)
]

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'  # noqa

# We only save the best checkpoint by validation mAP.
default_hooks = dict(
    checkpoint=dict(save_best='auto', max_keep_ckpts=-1, interval=-1))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)

broadcast_buffers = True
