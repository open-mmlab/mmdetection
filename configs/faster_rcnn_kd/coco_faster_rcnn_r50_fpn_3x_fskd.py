_base_ = [
    '../faster_rcnn/coco_faster_rcnn_r50_fpn_3x.py',
]


# model
model = dict(type='FasterRCNN_TS',
             distill_param=1.5,
             distill_param_backbone=1.5,
             roi_head=dict(
                 type='ContRoIHead'
                ),
            )


# Distillation Params
teacher_config_path = 'result/coco/faster_rcnn_r50_fpn_3x/coco_faster_rcnn_r50_fpn_3x.py'
teacher_weight_path = 'result/coco/faster_rcnn_r50_fpn_3x/epoch_36.pth'
backbone_pretrain = False


# Dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


pre_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
]

train_pipeline = [
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(type="CocoContDataset",
               pipeline=train_pipeline,
               pre_pipeline=pre_train_pipeline,
               multiscale_mode_student='range', # range
               ratio_hr_lr_student=0.5,
               min_lr_student=0.6),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))