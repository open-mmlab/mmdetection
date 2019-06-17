# model settings
input_size = 512
model = dict(
    type='SingleStageDetector',
    pretrained='open-mmlab://vgg16_caffe',
    backbone=dict(
        type='SSDVGG_M2Det',
        input_size=input_size,
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        l2_norm_scale=20),
    neck=dict(
        type='MLFPN',
        backbone_choice="SSD",  # "SSD" or "ResNet"
        in_channels=[512, 1024, 512, 256, 256, 256],
        planes=256,
        scale_outs_num=6,
        tum_num=8,
        smooth=True,
        base_feature_size=4,
        base_choice=2,
        base_list=[2, 3],
        norm=True,
        ssd_style_tum=False
        # the size of the smallest tum ouput =>( '-2' or '/2')
    ),
    bbox_head=dict(
        type='SSDHead',
        input_size=input_size,
        in_channels=(2048, 2048, 2048, 2048, 2048, 2048),
        num_classes=81,
        anchor_strides=(8, 16, 32, 64, 128, 256),
        basesize_ratio_range=(0.06, 1.05),
        anchor_ratios=([2, 3], [2, 3], [2, 3], [2, 3], [2], [2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2),
        ratio_change=True))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
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
dataset_type = 'CocoDataset'
data_root = '/data2/shengtao/COCO/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_scale=(512, 512),
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
                    ratio_range=(1, 4)),
                random_crop=dict(
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
            resize_keep_ratio=False)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=None,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[18, 24, 30])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 32
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/m2det512_vgg'
load_from = None
resume_from = None
workflow = [('train', 1)]
