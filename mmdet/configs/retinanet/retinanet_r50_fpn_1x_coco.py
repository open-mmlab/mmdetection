# Copyright (c) OpenMMLab. All rights reserved.
if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
    from .retinanet_tta import *

from torch.optim.sgd import SGD

# optimizer
optim_wrapper.update(
    dict(optimizer=dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0001)))
