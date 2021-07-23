import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class CheckIsfiniteHook(Hook):

    def __init__(
        self,
        interval=10,
        after_train_iter=True,
    ):
        self.interval = interval
        self._after_train_iter = after_train_iter

    def after_train_iter(self, runner):
        if self._after_train_iter and \
                self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                'loss become infinite or NaN!'
            print('loss is normal')
