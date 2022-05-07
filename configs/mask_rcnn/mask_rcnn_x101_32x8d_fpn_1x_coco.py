_base_ = './mask_rcnn_r101_fpn_1x_coco.py'
preprocess_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False,
    pad_size_divisor=32)
model = dict(
    # ResNeXt-101-32x8d model trained with Caffe2 at FB,
    # so the mean and std need to be changed.
    preprocess_cfg=preprocess_cfg,
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=8,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnext101_32x8d')))
