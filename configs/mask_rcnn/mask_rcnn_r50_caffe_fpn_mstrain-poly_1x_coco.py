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
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomChoiceResize',
        scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768),
               (1333, 800)]),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
