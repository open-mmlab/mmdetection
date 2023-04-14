if '_base_':
    from ..mask_rcnn.mask_rcnn_r50_caffe_fpn_1x_coco import *
from mmdet.models.detectors.mask_scoring_rcnn import MaskScoringRCNN
from mmdet.models.roi_heads.mask_scoring_roi_head import MaskScoringRoIHead
from mmdet.models.roi_heads.mask_heads.maskiou_head import MaskIoUHead

model.merge(
    dict(
        type=MaskScoringRCNN,
        roi_head=dict(
            type=MaskScoringRoIHead,
            mask_iou_head=dict(
                type=MaskIoUHead,
                num_convs=4,
                num_fcs=2,
                roi_feat_size=14,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                num_classes=80)),
        # model training and testing settings
        train_cfg=dict(rcnn=dict(mask_thr_binary=0.5))))
