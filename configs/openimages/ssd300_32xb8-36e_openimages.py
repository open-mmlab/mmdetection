_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/openimages_detection.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
model = dict(
    bbox_head=dict(
        num_classes=601,
        anchor_generator=dict(basesize_ratio_range=(0.2, 0.9))))
# dataset settings
dataset_type = 'OpenImagesDataset'
data_root = 'data/OpenImages/'
input_size = 300
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean={{_base_.model.data_preprocessor.mean}},
        to_rgb={{_base_.model.data_preprocessor.bgr_to_rgb}},
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]

train_dataloader = dict(
    batch_size=8,  # using 32 GPUS while training. total batch size is 32 x 8
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=3,  # repeat 3 times, total epochs are 12 x 3
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/oidv6-train-annotations-bbox.csv',
            data_prefix=dict(img='OpenImages/train/'),
            label_file='annotations/class-descriptions-boxable.csv',
            hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
            meta_file='annotations/train-image-metas.pkl',
            pipeline=train_pipeline)))
val_dataloader = dict(batch_size=8, dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(batch_size=8, dataset=dict(pipeline=test_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=5e-4))
# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=20000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=256)
