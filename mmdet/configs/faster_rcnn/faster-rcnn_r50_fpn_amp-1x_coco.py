# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper

with read_base():
    from .faster_rcnn_r50_fpn_1x_coco import *

optim_wrapper.update(dict(type=AmpOptimWrapper))
