_base_ = './knet_s3_r50_fpn_1x_coco-panoptic.py'

dataset_type = 'CocoPanopticDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]

# Use RepeatDataset to speed up training
data = dict(
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/panoptic_train2017.json',
            img_prefix=data_root + 'train2017/',
            seg_prefix=data_root + 'annotations/panoptic_train2017/',
            pipeline=train_pipeline)))
evaluation = dict(interval=1, metric=['pq'])
lr_config = dict(step=[9, 11])
