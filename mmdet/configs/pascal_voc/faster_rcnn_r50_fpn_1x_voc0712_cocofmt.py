if '_base_':
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.datasets.voc0712 import *
    from .._base_.default_runtime import *
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations, LoadAnnotations
from mmdet.datasets.transforms.transforms import Resize, RandomFlip, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs
from mmengine.dataset.dataset_wrapper import RepeatDataset
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmengine.runner.loops import EpochBasedTrainLoop, ValLoop, TestLoop
from mmengine.optim.scheduler.lr_scheduler import MultiStepLR
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD

model.merge(dict(roi_head=dict(bbox_head=dict(num_classes=20))))

METAINFO = {
    'classes':
    ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
     'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    # palette is a list of color tuples, which is used for visualization.
    'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252),
                (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0),
                (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]
}

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=Resize, scale=(1000, 600), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1000, 600), keep_ratio=True),
    # avoid bboxes being resized
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader.merge(
    dict(
        dataset=dict(
            type=RepeatDataset,
            times=3,
            dataset=dict(
                _delete_=True,
                type=dataset_type,
                data_root=data_root,
                ann_file='annotations/voc0712_trainval.json',
                data_prefix=dict(img=''),
                metainfo=METAINFO,
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
                backend_args=backend_args))))
val_dataloader.merge(
    dict(
        dataset=dict(
            type=dataset_type,
            ann_file='annotations/voc07_test.json',
            data_prefix=dict(img=''),
            metainfo=METAINFO,
            pipeline=test_pipeline)))
test_dataloader = val_dataloader

val_evaluator.merge(
    dict(
        type=CocoMetric,
        ann_file=data_root + 'annotations/voc07_test.json',
        metric='bbox',
        format_only=False,
        backend_args=backend_args))
test_evaluator = val_evaluator

# training schedule, the dataset is repeated 3 times, so the
# actual epoch = 4 * 3 = 12
max_epochs = 4
train_cfg = dict(
    type=EpochBasedTrainLoop, max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# learning rate
param_scheduler = [
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
