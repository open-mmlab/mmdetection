_base_ = './fast_rcnn_r50_caffe_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))
