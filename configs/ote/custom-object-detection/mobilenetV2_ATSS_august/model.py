_base_ = [
    './coco_data_pipeline.py'
]
model = dict(
    type='ATSS',
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(2, 3, 4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=64,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=80,
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[0.5, 1.0, 2.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

evaluation = dict(interval=1, metric_items=['mAP'], save_best='bbox_mAP')
optimizer = dict(
    type='SGD',
    lr=0.009,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='bbox_mAP',
    patience=3,
    iteration_patience=600,
    interval=1,
    min_lr=0.000009,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3)

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
runner = dict(type='EpochRunnerWithCancel', max_epochs=300)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'output'
load_from = None
resume_from = None
workflow = [('train', 1)]
custom_hooks = [
    dict(type='EarlyStoppingHook', patience=5, iteration_patience=1000, metric='bbox_mAP', interval=1, priority=75)
]