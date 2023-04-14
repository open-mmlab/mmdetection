if '_base_':
    from .faster_rcnn_r50_fpg_crop640_50e_coco import *

norm_cfg.merge(dict(type='BN', requires_grad=True))
model.merge(
    dict(
        neck=dict(out_channels=128, inter_channels=128),
        rpn_head=dict(in_channels=128),
        roi_head=dict(
            bbox_roi_extractor=dict(out_channels=128),
            bbox_head=dict(in_channels=128))))
