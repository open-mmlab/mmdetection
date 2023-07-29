# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0
from mmengine.config import read_base

with read_base():
    from .htc_without_semantic_r50_fpn_1x_coco import *

from mmcv.transforms import LoadImageFromFile

from mmdet.datasets.transforms import (LoadAnnotations, PackDetInputs,
                                       RandomFlip, Resize)
from mmdet.models.roi_heads.mask_heads.fused_semantic_head import \
    FusedSemanticHead

model.update(
    dict(
        data_preprocessor=dict(pad_seg=True),
        roi_head=dict(
            semantic_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    type=RoIAlign, output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8]),
            semantic_head=dict(
                type=FusedSemanticHead,
                num_ins=5,
                fusion_level=1,
                seg_scale_factor=1 / 8,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=183,
                loss_seg=dict(
                    type=CrossEntropyLoss, ignore_index=255,
                    loss_weight=0.2)))))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True, with_seg=True),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]
train_dataloader.update(
    dict(
        dataset=dict(
            data_prefix=dict(
                img='train2017/', seg='stuffthingmaps/train2017/'),
            pipeline=train_pipeline)))
