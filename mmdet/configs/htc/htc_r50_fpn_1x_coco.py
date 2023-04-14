if '_base_':
    from .htc_without_semantic_r50_fpn_1x_coco import *
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor
from mmdet.models.roi_heads.mask_heads.fused_semantic_head import FusedSemanticHead
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import Resize, RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs

model.merge(
    dict(
        data_preprocessor=dict(pad_seg=True),
        roi_head=dict(
            semantic_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    type='RoIAlign', output_size=14, sampling_ratio=0),
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
train_dataloader.merge(
    dict(
        dataset=dict(
            data_prefix=dict(
                img='train2017/', seg='stuffthingmaps/train2017/'),
            pipeline=train_pipeline)))
