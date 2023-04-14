if '_base_':
    from .retinanet_r50_caffe_fpn_ms_3x_coco import *
# learning policy
model.merge(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron2/resnet101_caffe'))))
