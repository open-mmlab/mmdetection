if '_base_':
    from .._base_.models.mask_rcnn_r50_fpn import *
    from .._base_.datasets.coco_instance import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmengine.visualization.vis_backend import LocalVisBackend

vis_backends = [dict(type=LocalVisBackend), dict(type='WandBVisBackend')]
visualizer.merge(dict(vis_backends=vis_backends))

# MMEngine support the following two ways, users can choose
# according to convenience
# default_hooks = dict(checkpoint=dict(interval=4))
default_hooks.checkpoint.interval = 4

# train_cfg = dict(val_interval=2)
train_cfg.val_interval = 2
