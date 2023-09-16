# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from ..common.ms_poly_3x_coco_instance import *
    from .._base_.models.mask_rcnn_r50_fpn import *

from mmengine.model.weight_init import PretrainedInit

model = dict(
    # use caffe img_norm
    data_preprocessor=dict(
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False),
    backbone=dict(
        depth=101,
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))
