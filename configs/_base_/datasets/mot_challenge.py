# dataset settings
dataset_type = 'MOTChallengeDataset'
data_root = 'data/MOT17/'
resized_shape = (1088, 1088)

# data pipeline
train_pipeline = [
    dict(
        type='UniformSample',
        num_ref_imgs=1,
        frame_range=10,
        filter_key_img=True),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations'),
            dict(
                type='RandomResize',
                scale=resized_shape,
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='PhotoMetricDistortion')
        ]),
    dict(
        type='TransformBroadcaster',
        # different coppped positions for different frames
        share_random_params=False,
        transforms=[
            dict(
                type='RandomCrop',
                crop_size=resized_shape,
                bbox_clip_border=False)
        ]),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs')
]

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=resized_shape, keep_ratio=True),
            dict(type='LoadTrackAnnotations')
        ]),
    dict(type='PackTrackInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    # MOTChallengeDataset is a video-based dataset, so we don't need
    # "AspectRatioBatchSampler"
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        visibility_thr=-1,
        ann_file='annotations/half-train_cocoformat.json',
        data_prefix=dict(img_path='train'),
        metainfo=dict(classes=('pedestrian', )),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='MOTChallengeMetric', metric=['HOTA', 'CLEAR', 'Identity'])
test_evaluator = val_evaluator
