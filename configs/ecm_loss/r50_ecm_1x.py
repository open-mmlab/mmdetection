_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

train_cfg = dict(val_interval=6)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1203,
            cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
            loss_cls=dict(
                type='EffectiveClassMarginLoss',
                use_sigmoid=True,
                fg_bg_ratio=6.3313,
                num_classes=1203,
                loss_weight=1.0,
            )),
        mask_head=dict(
            num_classes=1203,
            predictor_cfg=dict(type='NormedConv2d', tempearture=20))),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

# dataloader
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=8)
test_dataloader = dict(batch_size=4)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5, norm_type=2))
# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=False,
        begin=0,
        end=10000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
