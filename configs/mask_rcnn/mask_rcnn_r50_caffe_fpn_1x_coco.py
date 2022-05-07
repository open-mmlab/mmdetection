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
