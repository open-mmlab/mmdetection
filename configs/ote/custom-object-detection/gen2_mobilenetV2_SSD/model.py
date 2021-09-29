_base_ = [
    './coco_data_pipeline.py'
]
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
            widths=[[
                59.91402252688701, 100.44703428492618, 137.22888697945035,
                183.22823913364294
            ],
                    [
                        157.10841010152603, 203.7654599640001,
                        212.95826764946042, 317.3329515135652
                    ]],
            heights=[[
                40.06737185376501, 100.97103306538557, 74.35819037443196,
                73.83043437079853
            ],
                     [
                         108.31147481737037, 113.97849956820743,
                         183.05077289009338, 167.13243858925668
                     ]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2)),
        depthwise_heads=True,
        depthwise_heads_activations='relu',
        loss_balancing=True),
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
        min_bbox_size=0,
        score_thr=0.02,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=200))

cudnn_benchmark = True
evaluation = dict(interval=1000, metric='mAP')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.0001,
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=0.3333333333333333)

checkpoint_config = dict(interval=1000)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
runner = dict(type='IterBasedRunner', max_iters=10000)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'output'

#TODO: this is a temporary decision
import os
TT_API_OTHER_TESTS = os.environ.get('TT_API_OTHER_TESTS', '').lower() == 'true'
TT_PERFORMANCE_TESTS = os.environ.get('TT_PERFORMANCE_TESTS', '').lower() == 'true'
if TT_API_OTHER_TESTS or TT_PERFORMANCE_TESTS:
    load_from = '/mnt/ote_data/MODELS/mobilenetV2_SSD/coco_snapshot.pth'
    print(f'set load_from={load_from}', flush=True)
else:
    load_from = None
del os # this del is required for mmdetection config

resume_from = None
workflow = [('train', 1)]
