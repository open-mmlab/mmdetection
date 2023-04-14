if '_base_':
    from .._base_.models.ssd300 import *
    from .._base_.datasets.openimages_detection import *
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_1x import *
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations, LoadAnnotations
from mmdet.datasets.transforms.transforms import PhotoMetricDistortion, Expand, MinIoURandomCrop, Resize, RandomFlip, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs
from mmengine.dataset.dataset_wrapper import RepeatDataset
from torch.optim.sgd import SGD
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

model.merge(
    dict(
        bbox_head=dict(
            num_classes=601,
            anchor_generator=dict(basesize_ratio_range=(0.2, 0.9)))))
# dataset settings
dataset_type = 'OpenImagesDataset'
data_root = 'data/OpenImages/'
input_size = 300
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PhotoMetricDistortion,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type=Expand,
        mean=model.data_preprocessor.mean,
        to_rgb=model.data_preprocessor.bgr_to_rgb,
        ratio_range=(1, 4)),
    dict(
        type=MinIoURandomCrop,
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type=Resize, scale=(input_size, input_size), keep_ratio=False),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(input_size, input_size), keep_ratio=False),
    # avoid bboxes being resized
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]

train_dataloader.merge(
    dict(
        batch_size=8,  # using 32 GPUS while training. total batch size is 32 x 8
        batch_sampler=None,
        dataset=dict(
            _delete_=True,
            type=RepeatDataset,
            times=3,  # repeat 3 times, total epochs are 12 x 3
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='annotations/oidv6-train-annotations-bbox.csv',
                data_prefix=dict(img='OpenImages/train/'),
                label_file='annotations/class-descriptions-boxable.csv',
                hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
                meta_file='annotations/train-image-metas.pkl',
                pipeline=train_pipeline))))
val_dataloader.merge(dict(batch_size=8, dataset=dict(pipeline=test_pipeline)))
test_dataloader.merge(dict(batch_size=8, dataset=dict(pipeline=test_pipeline)))

# optimizer
optim_wrapper.merge(
    dict(optimizer=dict(type=SGD, lr=0.04, momentum=0.9, weight_decay=5e-4)))
# learning rate
param_scheduler = [
    dict(
        type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=20000),
    dict(
        type=MultiStepLR,
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (8 samples per GPU)
auto_scale_lr.merge(dict(base_batch_size=256))
