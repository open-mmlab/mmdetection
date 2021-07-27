_base_ = './yolox_s.py'

# model settings
# 1 depth=0.33, width=0.375
model = dict(
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, csp_num_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96)
)

img_norm_cfg = dict(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size=(416, 416), pad_val=114.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

# 2 close mixup, scale=(0.5, 1.5)
train_dataset = dict(
    mosaic_pipeline=[
        dict(type="RandomAffineOrPerspective", scale=(0.5, 1.5), border=(-320, -320))
    ],
    enable_mixup=False)

data = dict(
    train=train_dataset,
    test=dict(pipeline=test_pipeline),
    val=dict(pipeline=test_pipeline))

resume_from = None

interval = 10
evaluation = dict(interval=interval, metric='bbox')
# 3 random_size=(10, 20)
custom_hooks = [
    dict(
        type='YoloXProcessHook',
        random_size=(10, 20),
        no_aug_epoch=15,
        eval_interval=interval,
        priority=48),
    dict(type='EMAHook', priority=49, resume_from=resume_from)
]
log_config = dict(interval=50)
checkpoint_config = dict(interval=interval)

