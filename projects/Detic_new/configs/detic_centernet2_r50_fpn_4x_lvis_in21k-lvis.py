_base_ = './detic_centernet2_r50_fpn_4x_lvis_boxsup.py'
dataset_type = ['LVISV1Dataset', 'ImageNetLVISV1Dataset']
image_size_det = (640, 640)
image_size_cls = (320, 320)

# backend = 'pillow'
backend_args = None

train_pipeline_det = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=image_size_det,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size_det,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_pipeline_cls = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=False, with_label=True),
    dict(
        type='RandomResize',
        scale=image_size_cls,
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size_cls,
        recompute_bbox=False,
        bbox_clip_border=False,
        allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

dataset_det = dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type='LVISV1Dataset',
        data_root='data/lvis/',
        ann_file='annotations/lvis_v1_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline_det,
        backend_args=backend_args))

dataset_cls = dict(
    type='ImageNetLVISV1Dataset',
    data_root='data/imagenet',
    ann_file='annotations/imagenet_lvis_image_info.json',
    data_prefix=dict(img='ImageNet-LVIS/'),
    pipeline=train_pipeline_cls,
    backend_args=backend_args)

train_dataloader = dict(
    _delete_=True,
    batch_size=[8, 32],
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='MultiDataSampler', dataset_ratio=[1, 4]),
    batch_sampler=dict(
        type='MultiDataAspectRatioBatchSampler', num_datasets=2),
    dataset=dict(type='ConcatDataset', datasets=[dataset_det, dataset_cls]))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        by_epoch=False,
        T_max=90000,
    )
]

load_from = './first_stage/detic_centernet2_r50_fpn_4x_lvis_boxsup.pth'

find_unused_parameters = True
