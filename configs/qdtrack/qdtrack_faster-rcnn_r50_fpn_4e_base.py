_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py', '../_base_/default_runtime.py'
]

detector = _base_.model
detector.pop('data_preprocessor')

detector['backbone'].update(
    dict(
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
detector.rpn_head.loss_bbox.update(
    dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0))
detector.rpn_head.bbox_coder.update(dict(clip_border=False))
detector.roi_head.bbox_head.update(dict(num_classes=1))
detector.roi_head.bbox_head.bbox_coder.update(dict(clip_border=False))
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
    'faster_rcnn_r50_fpn_1x_coco-person/'
    'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
    # noqa: E501
)
del _base_.model

model = dict(
    type='QDTrack',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    detector=detector,
    track_head=dict(
        type='QuasiDenseTrackHead',
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        embed_head=dict(
            type='QuasiDenseEmbedHead',
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='MarginL2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))),
    tracker=dict(
        type='QuasiDenseTracker',
        init_score_thr=0.9,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=30,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'))
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning policy
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=4, by_epoch=True, milestones=[3])
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=4, val_interval=4)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]
