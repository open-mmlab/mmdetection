_base_ = [
    './coco_data_pipeline.py'
]
# model settings
width_mult = 1.0
model = dict(
    type='SingleStageDetector',
    pretrained=True,
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(4, 5),
        frozen_stages=-1,
        norm_eval=False),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        num_classes=80,
        in_channels=(96, 320),
        anchor_generator=dict(
            type='SSDAnchorGeneratorClustered',
            strides=(16, 32),
            widths=[
                [
                    11.777124212603184, 27.156337561336,
                    78.40999192363739, 42.895380750113695
                ],
                [
                    63.14842447887146, 115.46481026459409,
                    213.49145695359056, 138.2245536906473,
                    234.80364875556538
                ]
            ],
            heights=[
                [
                    14.767053135155848, 45.49947844712648,
                    45.981733925746965, 98.66743124119586
                ],
                [
                    177.24583777391308, 110.80317279721478,
                    95.85334315816411, 206.86475765838003,
                    220.30258590019886
                ]
            ]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2)),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=True),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.0,
        use_giou=False,
        use_focal=False,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))

evaluation = dict(interval=1000, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 3,
    step=[8000, 11000, 13000])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=13000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/mobilenet_v2-2s_ssd-256x256'
load_from = 'https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-256x256.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
