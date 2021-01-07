_base_ = [
    '../fcos/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco.py', '../_base_/swa.py'
]
model = dict(
    pretrained='open-mmlab://detectron/resnet101_caffe',
    backbone=dict(depth=101))
