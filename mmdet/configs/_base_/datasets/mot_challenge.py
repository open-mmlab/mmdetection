# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import (LoadImageFromFile, RandomResize,
                             TransformBroadcaster)

from mmdet.datasets import MOTChallengeDataset
from mmdet.datasets.samplers import TrackImgSampler
from mmdet.datasets.transforms import (LoadTrackAnnotations, PackTrackInputs,
                                       PhotoMetricDistortion, RandomCrop,
                                       RandomFlip, Resize,
                                       UniformRefFrameSample)
from mmdet.evaluation import MOTChallengeMetric

# dataset settings
dataset_type = MOTChallengeDataset
data_root = 'data/MOT17/'
img_scale = (1088, 1088)

backend_args = None
# data pipeline
train_pipeline = [
    dict(
        type=UniformRefFrameSample,
        num_ref_imgs=1,
        frame_range=10,
        filter_key_img=True),
    dict(
        type=TransformBroadcaster,
        share_random_params=True,
        transforms=[
            dict(type=LoadImageFromFile, backend_args=backend_args),
            dict(type=LoadTrackAnnotations),
            dict(
                type=RandomResize,
                scale=img_scale,
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type=PhotoMetricDistortion)
        ]),
    dict(
        type=TransformBroadcaster,
        # different cropped positions for different frames
        share_random_params=False,
        transforms=[
            dict(type=RandomCrop, crop_size=img_scale, bbox_clip_border=False)
        ]),
    dict(
        type=TransformBroadcaster,
        share_random_params=True,
        transforms=[
            dict(type=RandomFlip, prob=0.5),
        ]),
    dict(type=PackTrackInputs)
]

test_pipeline = [
    dict(
        type=TransformBroadcaster,
        transforms=[
            dict(type=LoadImageFromFile, backend_args=backend_args),
            dict(type=Resize, scale=img_scale, keep_ratio=True),
            dict(type=LoadTrackAnnotations)
        ]),
    dict(type=PackTrackInputs)
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=TrackImgSampler),  # image-based sampling
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
    # Now we support two ways to test, image_based and video_based
    # if you want to use video_based sampling, you can use as follows
    # sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    sampler=dict(type=TrackImgSampler),  # image-based sampling
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
    type=MOTChallengeMetric, metric=['HOTA', 'CLEAR', 'Identity'])
test_evaluator = val_evaluator
