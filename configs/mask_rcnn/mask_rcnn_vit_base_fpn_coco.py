_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model
model = dict(
    backbone=dict(
        _delete_=True,
        type='VisionTransformer',
        arch='b',
        drop_path_rate=0.1,
        out_indices=(2, 5, 8, 11),
        layer_cfgs=[
            dict(use_window=True, window_size=14),
            dict(use_window=True, window_size=14),
            dict(use_window=False, window_size=14),
            dict(use_window=True, window_size=14),
            dict(use_window=True, window_size=14),
            dict(use_window=False, window_size=14),
            dict(use_window=True, window_size=14),
            dict(use_window=True, window_size=14),
            dict(use_window=False, window_size=14),
            dict(use_window=True, window_size=14),
            dict(use_window=True, window_size=14),
            dict(use_window=False, window_size=14),
        ],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/mnt/lustre/share_data/liuyuan/work_dirs/mae_fp16_1600/epoch_1600_20220321-c2a7f905.pth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='MMSyncBN', requires_grad=True)),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            norm_cfg=dict(type='MMSyncBN', requires_grad=True)),
        mask_head=dict(norm_cfg=dict(type='MMSyncBN', requires_grad=True)),
    ))

# dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1024, 1024),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(1024, 1024)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline), samples_per_gpu=1)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'bias': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)
fp16 = dict(loss_scale=dict(init_scale=512))

# Learning rate schedule
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001)

# runner
runner = dict(type='EpochBasedRunner', max_epochs=50)

# runtime
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
