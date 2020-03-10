_base_ = [
    'component/rpn_r50_fpn.py', 'component/coco_detection.py',
    'component/schedule_1x.py', 'component/default_runtime.py'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes']),
]
data = dict(train=dict(pipeline=train_pipeline))
evaluation = dict(interval=1, metric='proposal_fast')
work_dir = './work_dirs/rpn_r50_fpn_1x'
