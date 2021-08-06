_base_ = [
    '../_base_/models/centernet2_cascade_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# # lr_config = dict(
# #     policy='step',
# #     warmup='linear',
# #     warmup_iters=5000,
# #     warmup_ratio=0.00001,
# #     step=[18])
# # runner = dict(type='EpochBasedRunner', max_epochs=20)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    )

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# add by jack for tersorboard
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

find_unused_parameters = True

workflow = [('train', 1), ('val', 1)]
