# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class FastStopTrainingHook(Hook):
    """Set runner's epoch information to the model."""

    def __init__(self, by_epoch, save_ckpt=False, stop_iter_or_epoch=5):
        self.by_epoch = by_epoch
        self.save_ckpt = save_ckpt
        self.stop_iter_or_epoch = stop_iter_or_epoch

    def after_train_iter(self, runner, batch_idx: int, data_batch: None,
                         outputs: None) -> None:
        if self.save_ckpt and self.by_epoch:
            # If it is epoch-based and want to save weights,
            # we must run at least 1 epoch.
            return
        if runner.iter >= self.stop_iter_or_epoch:
            raise RuntimeError('quick exit')

    def after_train_epoch(self, runner) -> None:
        if runner.epoch >= self.stop_iter_or_epoch - 1:
            raise RuntimeError('quick exit')
