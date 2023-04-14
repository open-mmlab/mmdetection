if '_base_':
    from ..mask_rcnn.mask_rcnn_r50_fpn_1x_coco import *
from mmdet.models.backbones.resnest import ResNeSt
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared4Conv1FCBBoxHead
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomChoiceResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs

norm_cfg = dict(type='SyncBN', requires_grad=True)
model.merge(
    dict(
        # use ResNeSt img_norm
        data_preprocessor=dict(
            mean=[123.68, 116.779, 103.939],
            std=[58.393, 57.12, 57.375],
            bgr_to_rgb=True),
        backbone=dict(
            type=ResNeSt,
            stem_channels=64,
            depth=50,
            radix=2,
            reduction_factor=4,
            avg_down_stride=True,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnest50')),
        roi_head=dict(
            bbox_head=dict(
                type=Shared4Conv1FCBBoxHead,
                conv_out_channels=256,
                norm_cfg=norm_cfg),
            mask_head=dict(norm_cfg=norm_cfg))))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(
        type=LoadAnnotations, with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type=RandomChoiceResize,
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

train_dataloader.merge(dict(dataset=dict(pipeline=train_pipeline)))
