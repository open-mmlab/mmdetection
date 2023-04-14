if '_base_':
    from .sparse_rcnn_r50_fpn_ms_480_800_3x_coco import *
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip, RandomCrop
from mmcv.transforms.wrappers import RandomChoice
from mmcv.transforms.processing import RandomChoiceResize, RandomChoiceResize, RandomChoiceResize
from mmdet.datasets.transforms.formatting import PackDetInputs

num_proposals = 300
model.merge(
    dict(
        rpn_head=dict(num_proposals=num_proposals),
        test_cfg=dict(
            _delete_=True, rpn=None, rcnn=dict(max_per_img=num_proposals))))

# augmentation strategy originates from DETR.
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomChoice,
        transforms=[[
            dict(
                type=RandomChoiceResize,
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type=RandomChoiceResize,
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type=RandomCrop,
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type=RandomChoiceResize,
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type=PackDetInputs)
]
train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
