_base_ = '../sparse_rcnn/coco_sparse_rcnn_r50_fpn_mstrain_480-800_3x.py'

model = dict(type='SparseRCNN_TS',
             roi_head=dict(
                type='ContSparseRoIHead',
             ))

# Distillation Params
teacher_config_path = 'result/coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x/coco_sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x.py'
teacher_weight_path = 'result/coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x/epoch_24.pth'
backbone_pretrain = False


img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)


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
               multiscale_mode_student='range',
               ratio_hr_lr_student=0.5,
               min_lr_student=0.6),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
