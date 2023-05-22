_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../_base_/default_runtime.py',
    '../_base_/datasets/semi_coco_detection.py'
]

detector = _base_.model
detector.bbox_head = dict(
    type='FAM3DHead',
    num_classes=20,
    in_channels=256,
    stacked_convs=4,
    feat_channels=256,
    anchor_type='anchor_based',
    anchor_generator=dict(
        type='AnchorGenerator',
        ratios=[1.0],
        octave_base_scale=8,
        scales_per_octave=1,
        strides=[8, 16, 32, 64, 128]),
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[.0, .0, .0, .0],
        target_stds=[0.1, 0.1, 0.2, 0.2]),
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        activated=True,  # use probability instead of logit as input
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0),
    loss_bbox=dict(type='GIoULoss', loss_weight=2.0))
detector.train_cfg = dict(
    assigner=dict(type='DynamicSoftLabelAssigner', topk=13, iou_weight=2.0),
    alpha=1,
    beta=6,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
detector.test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)

dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

batch_size = 5
num_workers = 5

labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/VOC2007/instances_trainval.json',
    data_prefix=dict(img='/VOC2007/JPEGImages/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32))
unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/VOC2012/instances_trainval.json',
    data_prefix=dict(img='/VOC2012/JPEGImages'),
    filter_cfg=dict(filter_empty_gt=False))

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 4]),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/VOC2007/instances_test.json',
        data_prefix=dict(img='/VOC2007/JPEGImages'),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

train_cfg = dict(type='IterBasedTrainLoop', max_iters=7200, val_interval=4000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=72000,
        by_epoch=False,
        milestones=[72000, 72000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, max_keep_ckpts=2))
log_processor = dict(by_epoch=False)

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='MeanTeacherHook', momentum=0.9995, interval=1),
]

auto_scale_lr = dict(enable=False, base_batch_size=16)
