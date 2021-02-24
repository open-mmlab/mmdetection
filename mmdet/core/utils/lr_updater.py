from mmcv.runner import HOOKS
from mmcv.runner.hooks.lr_updater import CosineAnealingLrUpdaterHook, annealing_cos

@HOOKS.register_module()
class CosineAnealingLrUntilEpochUpdaterHook(CosineAnealingLrUpdaterHook):
    """The same LR updater as CosineAnealing but with `last epoch` support.

    Args:
        last_epoch (int, optional): The number of the last epoch where LR
            updating stops. This value can be greater than total_epochs. If it
            is equal to -1 LR will update until the end of the training.
            Default: -1.
    """

    def __init__(self, last_epoch=-1, **kwargs):
        assert last_epoch != 0
        self.last_epoch = int(last_epoch)
        super(CosineAnealingLrUntilEpochUpdaterHook, self).__init__(**kwargs)
        if last_epoch > 0:
            assert self.by_epoch, '"last_epoch" requires "by_epoch" LR updating'

    def get_lr(self, runner, base_lr):
        if self.last_epoch == -1:
            self.last_epoch = runner.max_epochs

        if self.by_epoch:
            progress = runner.epoch
            max_progress = self.last_epoch
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if self.by_epoch and self.last_epoch < progress:
            return target_lr

        return annealing_cos(base_lr, target_lr, progress / max_progress)
