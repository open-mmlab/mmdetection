_base_ = ['./sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco.py']

model = dict(
    rpn_head=dict(
        _delete_=True,
        type='EmbeddingRPNHead',
        num_proposals=300,
        proposal_feature_channel=256,
    ), )
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
short_edge = (400, 500, 600)
min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, value) for value in short_edge],
        multiscale_mode='value',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(384, 600),
        allow_negative_crop=False),
    dict(
        type='Resize',
        img_scale=[(1333, value) for value in min_values],
        multiscale_mode='value',
        override=True,
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline), )
