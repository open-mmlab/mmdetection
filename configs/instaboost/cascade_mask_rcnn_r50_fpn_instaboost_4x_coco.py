_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.5,
        hflag=False,
        aug_ratio=0.5),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))
# learning policy
lr_config = dict(step=[32, 44])
runner = dict(type='EpochBasedRunner', max_epochs=48)
