# dataset settings
dataset_type = 'MOTChallengeDataset'
data_root = 'data/MOT17/'

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
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(
                type='RandomResize',
                scale=(1088, 1088),
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='PhotoMetricDistortion')
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
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs')
]

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_instance_id=True),
            dict(type='Resize', scale=(1088, 1088), keep_ratio=True)
        ]),
    dict(type='PackTrackInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
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
    sampler=dict(type='VideoSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/half-val_cocoformat.json',
        data_prefix=dict(img_path='train'),
        ref_img_sampler=None,
        load_as_video=True,
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='MOTChallengeMetrics', metric=['HOTA', 'CLEAR', 'Identity'])
test_evaluator = val_evaluator
