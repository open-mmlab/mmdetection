_base_ = './fcos_r50-caffe-fpn-gn-head_1x_coco.py'

# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet101_caffe')))
