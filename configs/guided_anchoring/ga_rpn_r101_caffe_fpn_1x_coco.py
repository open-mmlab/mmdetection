_base_ = './ga_rpn_r50_caffe_fpn_1x_coco.py'
# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))
