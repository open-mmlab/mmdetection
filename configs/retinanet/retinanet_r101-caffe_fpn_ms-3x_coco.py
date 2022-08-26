_base_ = './retinanet_r50-caffe_fpn_ms-3x_coco.py'
# learning policy
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101))
