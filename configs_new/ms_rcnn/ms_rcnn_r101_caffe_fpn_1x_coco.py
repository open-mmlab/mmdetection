_base_ = './ms_rcnn_r50_caffe_fpn_1x.py'
model = dict(
    pretrained='open-mmlab://resnet101_caffe', backbone=dict(depth=101))
work_dir = './work_dirs/ms_rcnn_r101_caffe_fpn_1x'
