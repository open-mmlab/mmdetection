_base_ = ['./fcos_r50_caffe_fpn_gn-head_4x4_2x_coco.py']
model = dict(
    pretrained='open-mmlab://detectron/resnet101_caffe',
    backbone=dict(depth=101))
