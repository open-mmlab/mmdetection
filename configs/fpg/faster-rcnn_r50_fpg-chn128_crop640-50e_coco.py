<<<<<<< HEAD
_base_ = 'faster-rcnn_r50_fpn_crop640-50e_coco.py'
=======
_base_ = 'faster-rcnn_r50_fpg_crop640-50e_coco.py'
>>>>>>> update config

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    neck=dict(out_channels=128, inter_channels=128),
    rpn_head=dict(in_channels=128),
    roi_head=dict(
        bbox_roi_extractor=dict(out_channels=128),
        bbox_head=dict(in_channels=128)))
