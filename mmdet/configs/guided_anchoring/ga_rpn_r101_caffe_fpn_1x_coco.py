if '_base_':
    from .ga_rpn_r50_caffe_fpn_1x_coco import *
# model settings
model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron2/resnet101_caffe'))))
