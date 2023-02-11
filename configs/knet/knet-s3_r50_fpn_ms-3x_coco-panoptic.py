_base_ = './knet-s3_r50_fpn_1x_coco-panoptic.py'

dataset_type = 'CocoPanopticDataset'
data_root = 'data/coco/'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadPanopticAnnotations'),
    dict(
        type='RandomResize', scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
# Use RepeatDataset to speed up training
data = dict(
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/panoptic_train2017.json',
            img_prefix=data_root + 'train2017/',
            seg_prefix=data_root + 'annotations/panoptic_train2017/',
            pipeline=train_pipeline)))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[9, 11],
        gamma=0.1)
]
