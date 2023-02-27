
_base_ = '../sparse_rcnn/coco_sparse_rcnn_r50_fpn_mstrain_480-800_3x.py'

model = dict(type='SparseRCNN_TS',
             roi_head=dict(
                type='ContSparseRoIHead',
             ))

# Distillation Params
teacher_config_path = 'result/coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x/coco_sparse_rcnn_r50_fpn_mstrain_480-800_3x.py'
teacher_weight_path = 'result/coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x/epoch_36.pth'
backbone_pretrain = False

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)


pre_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
]

train_pipeline = [
    dict(
        type='Resize',
        img_scale=[(1333, value) for value in min_values],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
    )


lr_config = dict(policy='step', step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)