if '_base_':
    from .._base_.models.fast_rcnn_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmcv.transforms.loading import LoadImageFromFile, LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadProposals, LoadAnnotations, LoadProposals
from mmdet.datasets.transforms.wrappers import ProposalBroadcaster, ProposalBroadcaster
from mmdet.datasets.transforms.transforms import Resize, RandomFlip, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadProposals, num_max_proposals=2000),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=ProposalBroadcaster,
        transforms=[
            dict(type=Resize, scale=(1333, 800), keep_ratio=True),
            dict(type=RandomFlip, prob=0.5),
        ]),
    dict(type=PackDetInputs)
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadProposals, num_max_proposals=None),
    dict(
        type=ProposalBroadcaster,
        transforms=[
            dict(type=Resize, scale=(1333, 800), keep_ratio=True),
        ]),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader.merge(
    dict(
        dataset=dict(
            proposal_file='proposals/rpn_r50_fpn_1x_train2017.pkl',
            pipeline=train_pipeline)))
val_dataloader.merge(
    dict(
        dataset=dict(
            proposal_file='proposals/rpn_r50_fpn_1x_val2017.pkl',
            pipeline=test_pipeline)))
test_dataloader = val_dataloader
