_base_ = './retinanet_r50_caffe_fpn_mstrain_3x_coco.py'
# learning policy
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101))
