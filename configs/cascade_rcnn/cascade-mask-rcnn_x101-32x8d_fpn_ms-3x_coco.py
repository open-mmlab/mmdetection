_base_ = './cascade-mask-rcnn_r50_fpn_ms-3x_coco.py'

model = dict(
    # ResNeXt-101-32x8d model trained with Caffe2 at FB,
    # so the mean and std need to be changed.
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
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
