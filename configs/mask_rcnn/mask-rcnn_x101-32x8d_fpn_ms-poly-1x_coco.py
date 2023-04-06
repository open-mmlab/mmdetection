_base_ = './mask-rcnn_r101_fpn_1x_coco.py'

model = dict(
    # ResNeXt-101-32x8d model trained with Caffe2 at FB,
    # so the mean and std need to be changed.
    data_preprocessor=dict(
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        bgr_to_rgb=False),
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

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
