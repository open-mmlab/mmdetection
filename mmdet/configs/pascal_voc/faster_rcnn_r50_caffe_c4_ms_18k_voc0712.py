if '_base_':
    from .._base_.models.faster_rcnn_r50_caffe_c4 import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.datasets.voc0712 import *
    from .._base_.default_runtime import *
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations, LoadAnnotations
from mmcv.transforms.processing import RandomChoiceResize
from mmdet.datasets.transforms.transforms import RandomFlip, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs
from mmengine.dataset.sampler import InfiniteSampler
from mmengine.dataset.dataset_wrapper import ConcatDataset
from mmdet.datasets.voc import VOCDataset, VOCDataset
from mmengine.runner.loops import IterBasedTrainLoop
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.sgd import SGD

model.merge(dict(roi_head=dict(bbox_head=dict(num_classes=20))))

# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=RandomChoiceResize,
        scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    # avoid bboxes being resized
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader.merge(
    dict(
        sampler=dict(type=InfiniteSampler, shuffle=True),
        dataset=dict(
            _delete_=True,
            type=ConcatDataset,
            datasets=[
                dict(
                    type=VOCDataset,
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline,
                    backend_args=backend_args),
                dict(
                    type=VOCDataset,
                    data_root=data_root,
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline,
                    backend_args=backend_args)
            ])))

val_dataloader.merge(dict(dataset=dict(pipeline=test_pipeline)))
test_dataloader = val_dataloader

# training schedule for 18k
max_iter = 18000
train_cfg.merge(
    dict(
        _delete_=True,
        type=IterBasedTrainLoop,
        max_iters=max_iter,
        val_interval=3000))

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[12000, 16000],
        gamma=0.1)
]

# optimizer
optim_wrapper.merge(
    dict(
        type=OptimWrapper,
        optimizer=dict(type=SGD, lr=0.02, momentum=0.9, weight_decay=0.0001)))

default_hooks.merge(dict(checkpoint=dict(by_epoch=False, interval=3000)))
log_processor.merge(dict(by_epoch=False))
