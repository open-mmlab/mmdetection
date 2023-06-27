# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        iter = runner.iter
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        model.epoch = epoch
        model.iter = iter
