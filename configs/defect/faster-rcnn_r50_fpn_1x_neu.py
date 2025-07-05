_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# --- Dataset settings ---
dataset_type = 'CocoDataset'
classes = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratch')

data_root = 'data/NEU/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    train=dict(
        type=dataset_type,
        img_prefix=data_root + 'train/',
        classes=classes,
        ann_file=data_root + 'annotations/train.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root + 'val/',
        classes=classes,
        ann_file=data_root + 'annotations/val.json'),
    test=dict(
        type=dataset_type,
        img_prefix=data_root + 'test/',
        classes=classes,
        ann_file=data_root + 'annotations/test.json')
)

evaluation = dict(interval=1, metric='bbox')

# --- Model settings ---
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes)
        )
    )
)

# --- Optimizer ---
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
