_base_ = './cascade_rcnn_r50_fpn_syncbn-all_mstrain_1x_coco.py'
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101))
