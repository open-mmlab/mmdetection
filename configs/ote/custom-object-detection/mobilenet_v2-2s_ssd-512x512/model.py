# _base_ = [
#     './coco_data_pipeline.py'
# ]
# model settings
width_mult = 1.0
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True),
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
                    23.554248425206367, 54.312675122672,
                    156.8199838472748, 85.79076150022739
                ],
                [
                    126.29684895774292, 230.92962052918818,
                    426.98291390718117, 276.4491073812946,
                    469.60729751113075
                ]],
            heights=[
                [
                    29.534106270311696, 90.99895689425296,
                    91.96346785149395, 197.3348624823917
                ],
                [
                    354.49167554782616, 221.60634559442957,
                    191.70668631632822, 413.72951531676006,
                    440.6051718003978
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
# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 3,
    step=[8, 11, 13])
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
runner = dict(meta=dict(exp_name='train'),
              max_epochs=30,
              type='EpochBasedRunner')
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'outputs/mobilenet_v2-2s_ssd-512x512'
load_from = 'https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-2s_ssd-512x512.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
