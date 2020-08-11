_base_ = './ga_rpn_r50_caffe_fpn_1x_coco.py'
# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101))
