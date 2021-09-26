_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../common/lsj_100e_coco_instance.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
# Use MMSyncBN that handles empty tensor in head. It can be changed to
# SyncBN after https://github.com/pytorch/pytorch/issues/36530 is fixed
# Requires MMCV-full after  https://github.com/open-mmlab/mmcv/pull/1205.
head_norm_cfg = dict(type='MMSyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_eval=False,
        norm_cfg=norm_cfg,
        init_cfg=None,
        style='caffe'),
    neck=dict(norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=head_norm_cfg),
        mask_head=dict(norm_cfg=head_norm_cfg)))

file_client_args = dict(backend='disk')
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
image_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Use RepeatDataset to speed up training
data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
