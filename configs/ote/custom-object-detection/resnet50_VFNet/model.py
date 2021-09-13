_base_ = [
    './coco_data_pipeline.py'
]
model = dict(
    type='VFNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

evaluation = dict(interval=1000, metric='mAP')
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=0.3333333333333333,
    step=[8000, 10000])
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
    load_from = '/usr/src/app/external/mmdetection/ote_data/MODELS/resnet50_VFNet/coco_snapshot.pth'
    print(f'set load_from={load_from}', flush=True)
else:
    load_from = None
del os # this del is required for mmdetection config

resume_from = None
workflow = [('train', 500)]
