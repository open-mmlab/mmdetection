_base_ = './mask_rcnn_r50_fpn_1x_coco.py'
preprocess_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False,
    pad_size_divisor=32)
model = dict(
    # use caffe img_norm
    preprocess_cfg=preprocess_cfg,
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    rpn_head=dict(
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        bbox_roi_extractor=dict(
            roi_layer=dict(
                type='RoIAlign',
                output_size=7,
                sampling_ratio=2,
                aligned=False)),
        bbox_head=dict(
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        mask_roi_extractor=dict(
            roi_layer=dict(
                type='RoIAlign',
                output_size=14,
                sampling_ratio=2,
                aligned=False))))
