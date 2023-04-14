if '_base_':
    from ..mask_rcnn.mask_rcnn_r50_caffe_fpn_ms_1x_coco import *
from mmdet.models.detectors.point_rend import PointRend
from mmdet.models.roi_heads.point_rend_roi_head import PointRendRoIHead
from mmdet.models.roi_heads.roi_extractors.generic_roi_extractor import GenericRoIExtractor
from mmdet.models.roi_heads.mask_heads.coarse_mask_head import CoarseMaskHead
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss, CrossEntropyLoss
from mmdet.models.roi_heads.mask_heads.mask_point_head import MaskPointHead
# model settings
model.merge(
    dict(
        type=PointRend,
        roi_head=dict(
            type=PointRendRoIHead,
            mask_roi_extractor=dict(
                type=GenericRoIExtractor,
                aggregation='concat',
                roi_layer=dict(
                    _delete_=True, type='SimpleRoIAlign', output_size=14),
                out_channels=256,
                featmap_strides=[4]),
            mask_head=dict(
                _delete_=True,
                type=CoarseMaskHead,
                num_fcs=2,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                num_classes=80,
                loss_mask=dict(
                    type=CrossEntropyLoss, use_mask=True, loss_weight=1.0)),
            point_head=dict(
                type=MaskPointHead,
                num_fcs=3,
                in_channels=256,
                fc_channels=256,
                num_classes=80,
                coarse_pred_each_layer=True,
                loss_point=dict(
                    type=CrossEntropyLoss, use_mask=True, loss_weight=1.0))),
        # model training and testing settings
        train_cfg=dict(
            rcnn=dict(
                mask_size=7,
                num_points=14 * 14,
                oversample_ratio=3,
                importance_sample_ratio=0.75)),
        test_cfg=dict(
            rcnn=dict(
                subdivision_steps=5,
                subdivision_num_points=28 * 28,
                scale_factor=2))))
