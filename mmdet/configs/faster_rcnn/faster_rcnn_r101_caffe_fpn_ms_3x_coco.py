if '_base_':
    from .faster_rcnn_r50_fpn_ms_3x_coco import *

model.merge(
    dict(
        backbone=dict(
            depth=101,
            norm_cfg=dict(requires_grad=False),
            norm_eval=True,
            style='caffe',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron2/resnet101_caffe'))))
