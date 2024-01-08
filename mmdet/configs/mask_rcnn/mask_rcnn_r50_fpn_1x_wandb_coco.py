# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .._base_.datasets.coco_instance import *
    from .._base_.default_runtime import *
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *

from mmengine.visualization import LocalVisBackend, WandbVisBackend

vis_backends.update(dict(type=WandbVisBackend))
vis_backends.update(dict(type=LocalVisBackend))
visualizer.update(dict(vis_backends=vis_backends))

# MMEngine support the following two ways, users can choose
# according to convenience
# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

default_hooks.update(dict(checkpoint=dict(interval=4)))

train_cfg.update(dict(val_interval=2))
