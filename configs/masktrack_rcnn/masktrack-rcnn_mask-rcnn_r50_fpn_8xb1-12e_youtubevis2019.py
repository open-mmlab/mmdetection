_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/youtube_vis.py', '../_base_/default_runtime.py'
]

detector = _base_.model
detector.pop('data_preprocessor')
detector.roi_head.bbox_head.update(dict(num_classes=40))
detector.roi_head.mask_head.update(dict(num_classes=40))
detector.train_cfg.rpn.sampler.update(dict(num=64))
detector.train_cfg.rpn_proposal.update(dict(nms_pre=200, max_per_img=200))
detector.train_cfg.rcnn.sampler.update(dict(num=128))
detector.test_cfg.rpn.update(dict(nms_pre=200, max_per_img=200))
detector.test_cfg.rcnn.update(dict(score_thr=0.01))
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'  # noqa: E501
)
del _base_.model

model = dict(
    type='MaskTrackRCNN',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    detector=detector,
    track_head=dict(
        type='RoITrackHead',
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        embed_head=dict(
            type='RoIEmbedHead',
            num_fcs=2,
            roi_feat_size=7,
            in_channels=256,
            fc_out_channels=1024),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=128,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    tracker=dict(
        type='MaskTrackRCNNTracker',
        match_weights=dict(det_score=1.0, iou=2.0, det_label=10.0),
        num_frames_retain=20))

dataset_type = 'YouTubeVISDataset'
data_root = 'data/youtube_vis_2019/'
dataset_version = data_root[-5:-1]  # 2019 or 2021

# train_dataloader
train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    batch_sampler=dict(type='TrackAspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2019_train.json',
        data_prefix=dict(img_path='train/JPEGImages'),
        pipeline=_base_.train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3.0,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# visualizer
default_hooks = dict(
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_begin=13)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# evaluator
val_evaluator = dict(
    type='YouTubeVISMetric',
    metric='youtube_vis_ap',
    outfile_prefix='./youtube_vis_results',
    format_only=True)
test_evaluator = val_evaluator

del detector
