if '_base_':
    from .fcos_r50_caffe_fpn_gn_head_1x_coco import *

# model settings
model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron/resnet101_caffe'))))
