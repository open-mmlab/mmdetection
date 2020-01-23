# model settings
input_size = 300
width_mult = 0.75
model = dict(
    type='SingleStageDetector',
    # The initial imagenet snapshot can be found in
    # https://github.com/tonylins/pytorch-mobilenet-v2
    pretrained='snapshots/mobilenet_v2.pth.tar',
    backbone=dict(
        type='SSDMobilenetV2',
        input_size=input_size,
        width_mult=width_mult,
        activation_type='relu',
        scales=1
        ),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        input_size=input_size,
        in_channels=(int(width_mult*480),),
        num_classes=2,
        anchor_strides=(16,),
        anchor_widths=([9.4, 25.1, 14.7, 34.7, 143.0,
                        77.4, 128.8, 51.1, 75.6],),
        anchor_heights=([15.0, 39.6, 25.5, 63.2, 227.5,
                         162.9, 124.5, 105.1, 72.6],),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=True))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.4,
        neg_iou_thr=0.4,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    use_giou=False,
    use_focal=False,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# model training and testing settings
# dataset settings
dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(input_size, input_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict
data = dict(
    imgs_per_gpu=65,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'train.txt',
            ],
            min_size=17,
            img_prefix=[data_root + 'WIDER_train/'],
            img_scale=(input_size, input_size),
            img_norm_cfg=img_norm_cfg,
            size_divisor=None,
            flip_ratio=0.5,
            with_mask=False,
            with_crowd=False,
            with_label=True,
            test_mode=False,
            extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                expand=dict(
                    mean=img_norm_cfg['mean'],
                    to_rgb=img_norm_cfg['to_rgb'],
                    ratio_range=(1, 2),
                    prob=0.0),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.1)),
            resize_keep_ratio=False)),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/val.txt',
        img_prefix=data_root + 'WIDER_val/',
        test_mode=True,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.5*1e-1, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 3,
    step=[40, 55, 65])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 70
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './models/wider_face_ssd300_mobilenet'
load_from = None
resume_from = None
workflow = [('train', 1)]
