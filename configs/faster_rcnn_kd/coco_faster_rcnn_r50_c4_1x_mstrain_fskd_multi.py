_base_ = '../faster_rcnn/coco_faster_rcnn_r50_c4_1x_mstrain.py'

# model
model = dict(type='FasterRCNN_TS',
             distill_param=1.0,
             roi_head=dict(
                 type='ContRoIHead'
                )
            )

# Distillation Params
teacher_config_path = 'result/coco/faster_rcnn_r50_c4_1x_mstrain/coco_faster_rcnn_r50_c4_1x_mstrain.py'
teacher_weight_path = 'result/coco/faster_rcnn_r50_c4_1x_mstrain/epoch_12.pth'
backbone_pretrain = False

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

pre_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
]

train_pipeline = [
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(type="CocoContDataset",
               pipeline=train_pipeline,
               pre_pipeline=pre_train_pipeline,
               multiscale_mode_student='range', # range
               ratio_hr_lr_student=0.5,
               min_lr_student=0.6)
    )