_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained=None,
    backbone=dict(
        type='ResNeSt',
        stem_channels=64,
        depth=50,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            num_classes=3,
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg,
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=3, # number of class
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    test_cfg=dict(
       rpn=dict(
           nms_pre=1000,
           max_per_img=1000,
           nms=dict(type='nms', iou_threshold=0.7),
           min_bbox_size=0),
       rcnn=dict(
           score_thr=0.05,
           nms=dict(type='nms', iou_threshold=0.5),
           max_per_img=200,
           mask_thr_binary=0.5)))

img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 1333), (1280, 1280), (1024, 1024)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', direction=['horizontal', 'vertical'], flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
        img_scale=[(1333, 1333), (1280, 1280), (1024, 1024)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip',direction=['horizontal','vertical']),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, # batch_size
    train=dict(pipeline=train_pipeline,
               type='CocoDataset',
               ann_file='../data/livecell_images/competition_coco_train_v2.json',
               img_prefix='../data/livecell_images/livecell_images/',
               classes=('shsy5y','astro','cort',)),
    val=dict(pipeline=test_pipeline,
               ann_file='../data/livecell_images/competition_coco_val_v2.json',
               img_prefix='../data/livecell_images/livecell_images/',
               classes=('shsy5y','astro','cort',)),
    test=dict(pipeline=test_pipeline),
               ann_file='../data/livecell_images/competition_coco_val_v2.json',
               img_prefix='../data/livecell_images/livecell_images/',
               classes=('shsy5y','astro','cort',))

evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
optimizer = dict(type='SGD', lr=0.005/8, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
runner = dict(max_epochs=20)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 20

load_from = './checkpoints/mask_rcnn_r50_fpn_20e_compet_epoch_10.pth'
data_root = '../data/livecell_images'
seed=0
gpu_ids = range(0, 1)
work_dir = '/kaggle/temp/'








