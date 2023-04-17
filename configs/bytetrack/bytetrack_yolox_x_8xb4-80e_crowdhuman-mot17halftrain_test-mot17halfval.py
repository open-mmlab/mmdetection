_base_ = ['../yolox/yolox_x_8xb8-300e_coco.py']

dataset_type = 'MOTChallengeDataset'
data_root = 'data/MOT17/'

img_scale = (800, 1440)  # w, h
batch_size = 4

detector = _base_.model
detector.pop('data_preprocessor')
detector.bbox_head.update(dict(num_classes=1))
detector.test_cfg.nms.update(dict(iou_threshold=0.7))
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'  # noqa: E501
)
del _base_.model

model = dict(
    type='ByteTrack',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        use_det_processor=True,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(576, 1024),
                size_divisor=32,
                interval=10)
        ]),
    detector=detector,
    tracker=dict(
        type='ByteTracker',
        motion=dict(type='KalmanFilter'),
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=False),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=img_scale, keep_ratio=True),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadTrackAnnotations'),
        ]),
    dict(type='PackTrackInputs')
]
train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='CocoDataset',
                    data_root='data/MOT17',
                    ann_file='annotations/half-train_cocoformat.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
                dict(
                    type='CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_train.json',
                    data_prefix=dict(img='train'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
                dict(
                    type='CocoDataset',
                    data_root='data/crowdhuman',
                    ann_file='annotations/crowdhuman_val.json',
                    data_prefix=dict(img='val'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian')),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
            ]),
        pipeline=train_pipeline))

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    # sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    sampler=dict(type='TrackImgSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
# default 8 gpu
base_lr = 0.001 / 2 * batch_size
optim_wrapper = dict(optimizer=dict(lr=base_lr))

# some hyper parameters
# training settings
total_epochs = 80
num_last_epochs = 10
resume_from = None
interval = 5

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=interval)

# learning policy
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 1 to 70 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0,
        T_max=total_epochs - num_last_epochs,
        end=total_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 10 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=total_epochs - num_last_epochs,
        end=total_epochs,
    )
]

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

default_hooks = dict(
    checkpoint=dict(_delete_=True, type='CheckpointHook', interval=interval),
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# evaluator
val_evaluator = dict(
    _delete_=True,
    type='MOTChallengeMetric',
    metric=['HOTA', 'CLEAR', 'Identity'],
    postprocess_tracklet_cfg=[
        dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
    ])
test_evaluator = val_evaluator
del detector
del _base_.tta_model
del _base_.img_scales
del _base_.tta_pipeline
del _base_.train_dataset
