_base_ = [
    './retinanet_r50_fpns4_1x_coco_sm_tinyperson.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='ScaleMatchResize', scale_match_type="MonotonicityScaleMatch",
         src_anno_file="data/coco/annotations/instances_train2017.json",
         dst_anno_file="data/tiny_set/mini_annotations/tiny_set_train_all_erase.json",
         bins=100, default_scale=0.25, scale_range=(0.1, 1)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(train=dict(pipeline=train_pipeline))
