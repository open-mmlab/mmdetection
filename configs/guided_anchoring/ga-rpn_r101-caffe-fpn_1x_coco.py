_base_ = './ga-rpn_r50-caffe-fpn_1x_coco.py'
# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))
