from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.lr_updater import (CosineAnnealingLrUpdaterHook,
                                          annealing_cos)


@HOOKS.register_module()
class YOLOXLrUpdaterHook(CosineAnnealingLrUpdaterHook):
    """YOLOX learning rate scheme.

    There are two main differences between YOLOXLrUpdaterHook
    and CosineAnnealingLrUpdaterHook.

       1. When the current running epoch is greater than
           `max_epoch-last_epoch`, a fixed learning rate will be used
       2. The exp warmup scheme is different with LrUpdaterHook in MMCV

    Args:
        num_last_epochs (int): The number of epochs with a fixed learning rate
           before the end of the training.
        warmup_ratio (float): LR used at the beginning of warmup.
           This parameter does not depend on the number of GPUs, so we need
           to multiply by work_size.
    """

    def __init__(self, num_last_epochs, **kwargs):
        _, self.work_size = get_dist_info()
        self.num_last_epochs = num_last_epochs
        super(YOLOXLrUpdaterHook, self).__init__(**kwargs)

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            # exp warmup scheme
            k = self.warmup_ratio * pow(
                (cur_iters + 1) / float(self.warmup_iters), 2)
            warmup_lr = [_lr * k * self.work_size for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def get_lr(self, runner, base_lr):
        last_iter = len(runner.data_loader) * self.num_last_epochs

        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        progress += 1

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio * self.work_size
        else:
            target_lr = self.min_lr * self.work_size

        if progress >= max_progress - last_iter:
            # fixed learning rate
            return target_lr
        else:
            return annealing_cos(
                base_lr * self.work_size, target_lr,
                (progress - self.warmup_iters) /
                (max_progress - self.warmup_iters - last_iter))
