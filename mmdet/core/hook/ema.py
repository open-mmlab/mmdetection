from mmcv.runner.hooks import HOOKS, Hook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import torch.nn as nn
import math
from mmdet.apis.train import MyMMDistributedDataParallel


def is_parallel(model):
    return type(model) in (
    nn.parallel.DataParallel, nn.parallel.DistributedDataParallel, MMDataParallel, MMDistributedDataParallel, MyMMDistributedDataParallel)


@HOOKS.register_module(force=True)
class EMAHook(Hook):
    r"""Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointHook.

    Args:
        decay (float): Exponential decay coefficient. Default to 0.9998
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        resume_from (str, Optional): The checkpoint path. Defaults to None.
    """
    def __init__(self, decay=0.9998, interval=1, resume_from=None):
        self.interval = interval
        self.checkpoint = resume_from
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.

        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_parallel(model):
            model = model.module
        self.param_ema_buffer = {}
        self.model_parameters = model.state_dict()  # BN also needs ema
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        curr_step = runner.iter
        decay = self.decay(curr_step)
        if curr_step % self.interval != 0:
            return
        for name, parameter in self.model_parameters.items():
            # exclude num_tracking
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(decay).add_(parameter.data, alpha=1 - decay)

    def after_train_epoch(self, runner):
        """We load parameter values from ema backup to model before the
        EvalHook."""
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        """We recover model's parameter from ema backup after last epoch's
        EvalHook."""
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
