from mmcv.runner.hooks.lr_updater import CosineAnnealingLrUpdaterHook, annealing_cos
from mmcv.runner.hooks import HOOKS


@HOOKS.register_module()
class CosineAnnealingWithNoAugIterLrUpdaterHook(CosineAnnealingLrUpdaterHook):

    def __init__(self, no_aug_epochs, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.no_aug_epochs = no_aug_epochs
        super(CosineAnnealingLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):

        no_aug_iter = len(runner.data_loader) * self.no_aug_epochs

        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress >= max_progress - no_aug_iter:
            return self.min_lr
        else:
            return annealing_cos(base_lr, target_lr,
                                 (progress - self.warmup_iters) / (max_progress - self.warmup_iters - no_aug_iter))
