# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .mask_rcnn_r50_fpn_1x_coco import *

from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

optim_wrapper.update(dict(type=AmpOptimWrapper))
