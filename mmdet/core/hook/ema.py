from mmcv.runner.hooks import HOOKS, Hook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import torch.nn as nn
import math

def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel, nn.parallel.DistributedDataParallel, MMDataParallel, MMDistributedDataParallel)

class BaseEMAHook(Hook):
    r"""Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointHook. Note,
    the original model parameters are actually saved in ema field after train.

    Args:
        skip_bn_running_stats (bool): Whether to skip the batchnorm running stats
            (running_mean, running_var), it does not perform the ema operation.
            Default to False.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        resume_from (str, Optional): The checkpoint path. Defaults to None.
    """

    def __init__(self, skip_bn_running_stats=False, interval=1, resume_from=None):
        self.skip_bn_running_stats = skip_bn_running_stats
        self.interval = interval
        self.checkpoint = resume_from

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.
        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_parallel(model):
            model = model.module
        self.param_ema_buffer = {}
        if self.skip_bn_running_stats:
            self.model_parameters = dict(model.named_parameters(recurse=True))
        else:
            self.model_parameters = model.state_dict()
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def get_momentum(self, runner):
        raise NotImplementedError

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if runner.iter % self.interval != 0:
            return
        momentum = self.get_momentum(runner)
        for name, parameter in self.model_parameters.items():
            # exclude num_tracking
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(momentum).add_(parameter.data, alpha=1 - momentum)

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

@HOOKS.register_module()
class ExpDecayEMAHook(BaseEMAHook):
    """ Exponential decay EMAHook.
    Args:
        decay (float): Exponential decay coefficient. Default to 0.9998
        total_iter (int): The total number of iterations of EMA decay.
           Defaults to 100.
    """
    def __init__(self, decay=0.9998, total_iter=2000, **kwargs):
        self.decay_fun = lambda x: decay * (1 - math.exp(-(x+1) / total_iter))
        super(ExpDecayEMAHook, self).__init__(**kwargs)

    def get_momentum(self, runner):
        return self.decay_fun(runner.iter)

@HOOKS.register_module()
class LinerDecayEMAHook(BaseEMAHook):
    """ Liner decay EMAHook.
    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        warm_up (int): During first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly. Defaults to 100.
    """
    def __init__(self, momentum=0.0002, warm_up=100, **kwargs):
        assert 0 < momentum < 1
        super(LinerDecayEMAHook, self).__init__(**kwargs)
        self.decay_fun = lambda x: 1-min(momentum ** self.interval, (1 + x) / (warm_up + x))

    def get_momentum(self, runner):
        return self.decay_fun(runner.iter)
