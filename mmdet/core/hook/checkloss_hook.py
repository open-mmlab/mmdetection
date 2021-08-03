import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check Invalid Loss Hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 10.
        after_train_iter (bool): Whether check invalid loss in
            `after_train_iter`. Default: True.
    """

    def __init__(self, interval=10, after_train_iter=True):
        self.interval = interval
        self._after_train_iter = after_train_iter

    def after_train_iter(self, runner):
        if self._after_train_iter and \
                self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')
