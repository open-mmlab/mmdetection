_base_ = 'ssd300_voc0712.py'

input_size = 512
model = dict(
    neck=dict(
        out_channels=(512, 1024, 512, 256, 256, 256, 256),
        level_strides=(2, 2, 2, 2, 1),
        level_paddings=(1, 1, 1, 1, 1),
        last_kernel_size=4),
    bbox_head=dict(
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        anchor_generator=dict(
            input_size=input_size,
            strides=[8, 16, 32, 64, 128, 256, 512],
            basesize_ratio_range=(0.15, 0.9),
            ratios=([2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]))))

# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
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
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(input_size, input_size), keep_ratio=False),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=3,
    dataset=dict(  # RepeatDataset
        # the dataset is repeated 10 times, and the training schedule is 2x,
        # so the actual epoch = 12 * 10 = 120.
        times=10,
        dataset=dict(  # ConcatDataset
            # VOCDataset will add different `dataset_type` in dataset.metainfo,
            # which will get error if using ConcatDataset. Adding
            # `ignore_keys` can avoid this error.
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline),
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline)
            ])))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
