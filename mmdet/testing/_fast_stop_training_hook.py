# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class FastStopTrainingHook(Hook):
    """Set runner's epoch information to the model."""

    def after_train_iter(self, runner, batch_idx: int, data_batch: None,
                         outputs: None) -> None:
        if batch_idx >= 5:
            raise RuntimeError('quick exit')
