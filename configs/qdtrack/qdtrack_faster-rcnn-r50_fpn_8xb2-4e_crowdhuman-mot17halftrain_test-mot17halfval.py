_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_'
    'test-mot17halfval.py'
]

# data pipeline
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(
                type='mmdet.RandomResize',
                resize_type='mmdet.Resize',
                scale=(1088, 1088),
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='mmdet.PhotoMetricDistortion')
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(
                type='RandomCrop',
                crop_size=(1088, 1088),
                bbox_clip_border=False)
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='mmdet.RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs', ref_prefix='ref')
]
mot_cfg = dict(
    type='MOTChallengeDataset',
    data_root='data/MOT17',
    metainfo=dict(classes=('pedestrian')),
    visibility_thr=-1,
    ann_file='annotations/half-train_cocoformat.json',
    data_prefix=dict(img_path='train'),
    ref_img_sampler=dict(
        num_ref_imgs=1, frame_range=10, filter_key_img=True, method='uniform'),
    pipeline=train_pipeline)
crowdhuman_cfg = dict(
    type='BaseVideoDataset',
    data_root='data/crowdhuman',
    load_as_video=False,
    metainfo=dict(classes=('pedestrian')),
    ann_file='annotations/crowdhuman_train.json',
    data_prefix=dict(img_path='train'),
    ref_img_sampler=dict(num_ref_imgs=1, frame_range=0),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[mot_cfg, crowdhuman_cfg]))
