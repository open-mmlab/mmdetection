_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    rpn_head=dict(
        anchor_generator=dict(type='LegacyAnchorGenerator', center_offset=0.5),
        bbox_coder=dict(type='LegacyDeltaXYWHBBoxCoder'),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=2,
                aligned=False)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=14,
                sampling_ratio=2,
                aligned=False)),
        bbox_head=dict(
            bbox_coder=dict(type='LegacyDeltaXYWHBBoxCoder'),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),

    # model training and testing settings
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=2000),
        rcnn=dict(assigner=dict(match_low_quality=True))))
