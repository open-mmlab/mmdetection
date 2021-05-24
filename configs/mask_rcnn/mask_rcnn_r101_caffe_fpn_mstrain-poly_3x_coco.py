_base_ = [
    '../common/mstrain-poly_3x_coco_instance.py',
    '../_base_/models/mask_rcnn_r50_fpn.py'
]

model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(
        norm_cfg=dict(depth=101, requires_grad=False),
        norm_eval=True,
        style='caffe'))
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
