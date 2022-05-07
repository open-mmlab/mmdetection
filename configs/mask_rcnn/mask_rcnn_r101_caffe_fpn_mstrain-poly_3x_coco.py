_base_ = [
    '../common/mstrain-poly_3x_coco_instance.py',
    '../_base_/models/mask_rcnn_r50_fpn.py'
]

preprocess_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False,
    pad_size_divisor=32)
model = dict(
    # use caffe img_norm
    preprocess_cfg=preprocess_cfg,
    backbone=dict(
        depth=101,
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))
