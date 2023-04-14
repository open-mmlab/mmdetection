if '_base_':
    from ..faster_rcnn.faster_rcnn_r50_fpn_1x_coco import *
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor

model.merge(
    dict(
        roi_head=dict(
            bbox_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    _delete_=True,
                    type='DeformRoIPoolPack',
                    output_size=7,
                    output_channels=256),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]))))
