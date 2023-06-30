dataset_type = 'YouTubeVISDataset'
data_root = 'data/youtube_vis_2019/'
dataset_version = data_root[-5:-1]  # 2019 or 2021

backend_args = None

# dataset settings
train_pipeline = [
    dict(
        type='UniformRefFrameSample',
        num_ref_imgs=1,
        frame_range=100,
        filter_key_img=True),
    dict(
        type='TransformBroadcaster',
        share_random_params=True,
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadTrackAnnotations', with_mask=True),
            dict(type='Resize', scale=(640, 360), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackTrackInputs')
]

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(640, 360), keep_ratio=True),
            dict(type='LoadTrackAnnotations', with_mask=True),
        ]),
    dict(type='PackTrackInputs')
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    # sampler=dict(type='TrackImgSampler'),  # image-based sampling
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='TrackAspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2019_train.json',
        data_prefix=dict(img_path='train/JPEGImages'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2019_valid.json',
        data_prefix=dict(img_path='valid/JPEGImages'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
