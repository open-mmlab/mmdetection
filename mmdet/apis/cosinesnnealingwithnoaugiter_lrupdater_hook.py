from mmcv.runner.hooks.lr_updater import CosineAnnealingLrUpdaterHook, annealing_cos
from mmcv.runner.hooks import HOOKS
from mmcv.runner import get_dist_info


@HOOKS.register_module()
class CosineAnnealingWithNoAugIterLrUpdaterHook(CosineAnnealingLrUpdaterHook):

    def __init__(self, no_aug_epochs, min_lr=None, min_lr_ratio=None, warmup_ratio=0.1, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        _, work_size = get_dist_info()
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.no_aug_epochs = no_aug_epochs
        self.base_lr = warmup_ratio * work_size
        super(CosineAnnealingLrUpdaterHook, self).__init__(warmup_ratio=warmup_ratio * work_size, **kwargs)

    def get_warmup_lr(self, cur_iters):
        def _get_warmup_lr(cur_iters, regular_lr):
            k = self.warmup_ratio * pow(
                cur_iters / float(self.warmup_iters), 2
            )
            warmup_lr = [k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def get_lr(self, runner, base_lr):

        no_aug_iter = len(runner.data_loader) * self.no_aug_epochs

        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_lr_ratio is not None:
            target_lr = self.base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress >= max_progress - no_aug_iter:
            return self.min_lr
        else:
            return annealing_cos(self.base_lr, target_lr,
                                 (progress - self.warmup_iters) / (max_progress - self.warmup_iters - no_aug_iter))
