from mmengine.config import read_base

with read_base():
    from mmdet.configs.dino.dino_5scale_swin_l_8xb2_12e_coco import *  # noqa

from projects.example_largemodel import (checkpoint_check_fn,
                                         layer_auto_wrap_policy)

# The checkpoint needs to be controlled by the checkpoint_check_fn.
model.update(dict(backbone=dict(with_cp=False)))  # noqa

# TODO: The new version of configs does not support passing a module list,
#  so for now, it can only be hard-coded. We will fix this issue in the future.
runner_type = 'FlexibleRunner'
strategy = dict(
    type='FSDPStrategy',
    activation_checkpointing=dict(check_fn=checkpoint_check_fn),
    model_wrapper=dict(auto_wrap_policy=dict(type=layer_auto_wrap_policy)))
