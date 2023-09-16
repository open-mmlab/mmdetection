# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .mask_rcnn_r18_fpn_8xb8_amp_lsj_200e_coco import *

from mmengine.model.weight_init import PretrainedInit

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type=PretrainedInit, checkpoint='torchvision://resnet101')))
