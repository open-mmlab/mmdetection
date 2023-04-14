if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.coco_instance import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.backbones.regnet import RegNet
from mmdet.models.necks.fpn import FPN
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomChoiceResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs
from torch.optim.sgd import SGD
from mmengine.optim.scheduler.lr_scheduler import LinearLR, MultiStepLR

model.merge(
    dict(
        data_preprocessor=dict(
            # The mean and std are used in PyCls when training RegNets
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            bgr_to_rgb=False),
        backbone=dict(
            _delete_=True,
            type=RegNet,
            arch='regnetx_3.2gf',
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://regnetx_3.2gf')),
        neck=dict(
            type=FPN,
            in_channels=[96, 192, 432, 1008],
            out_channels=256,
            num_outs=5)))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(
        type=RandomChoiceResize,
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))

optim_wrapper.merge(
    dict(
        optimizer=dict(type=SGD, lr=0.02, momentum=0.9, weight_decay=0.00005),
        clip_grad=dict(max_norm=35, norm_type=2)))

# learning policy
max_epochs = 36
train_cfg.merge(dict(max_epochs=max_epochs))
param_scheduler = [
    dict(type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[28, 34],
        gamma=0.1)
]
