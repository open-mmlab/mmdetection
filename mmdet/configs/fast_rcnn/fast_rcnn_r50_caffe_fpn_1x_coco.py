if '_base_':
    from .fast_rcnn_r50_fpn_1x_coco import *
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor

model.merge(
    dict(
        data_preprocessor=dict(
            type=DetDataPreprocessor,
            mean=[103.530, 116.280, 123.675],
            std=[1.0, 1.0, 1.0],
            bgr_to_rgb=False,
            pad_size_divisor=32),
        backbone=dict(
            norm_cfg=dict(type='BN', requires_grad=False),
            style='caffe',
            norm_eval=True,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron2/resnet50_caffe'))))
