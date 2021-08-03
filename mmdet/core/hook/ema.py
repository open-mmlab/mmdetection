import math

from mmcv.runner.hooks import HOOKS, Hook
from mmcv.parallel import is_module_wrapper

class BaseEMAHook(Hook):
    """Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointHook. Note,
    the original model parameters are actually saved in ema field after train.

    Args:
        decay (float): The decay used for updating ema parameter.
            Defaults to 0.9998. Ema paramters are updated with the formula:
            `ema_param = decay * ema_param + (1 - decay) * cur_param`
        skip_buffers (bool): Whether to skip the model buffers, such as batchnorm running
            stats (running_mean, running_var), it does not perform the ema operation.
            Default to False.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        resume_from (str, Optional): The checkpoint path. Defaults to None.
        decay_fun (func, Optional): the function to change decay during early iteration
            (also warmup) to help early training. Defaults to None, it use `decay` as a
            constant.
    """

    def __init__(self, decay=0.9998, skip_buffers=False, interval=1,
        resume_from=None, decay_fun=None):
        assert 0 < decay < 1
        self.decay = decay
        self.skip_buffers = skip_buffers
        self.interval = interval
        self.checkpoint = resume_from
        self.decay_fun = decay_fun

    def before_run(self, runner):
        """To resume model with it's ema parameters more friendly.
        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        if self.skip_buffers:
            self.model_parameters = dict(model.named_parameters())
        else:
            self.model_parameters = model.state_dict()
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers())
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def get_decay(self, runner):
        return self.decay_fun(runner.iter) if self.decay_fun else \
                        self.decay

    def after_train_iter(self, runner):
        """Update ema parameter every self.interval iterations."""
        if runner.iter % self.interval != 0:
            return
        decay = self.get_decay(runner)
        for name, parameter in self.model_parameters.items():
            # exclude num_tracking
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(decay).add_(parameter.data, alpha=1-decay)

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
    """EMAHook using exponential decay strategy

    Args:
        total_iter (int): The total number of iterations of EMA decay.
           Defaults to 2000.
    """
    def __init__(self, total_iter=2000, **kwargs):
        super(ExpDecayEMAHook, self).__init__(**kwargs)
        self.decay_fun = lambda x: self.decay * (1. - 
                                math.exp(-(x+1) / total_iter))
        

@HOOKS.register_module()
class LinearDecayEMAHook(BaseEMAHook):
    """EMAHook using linear decay strategy

    Args:
        warm_up (int): During first warm_up steps, we may use smaller decay
            to update ema parameters more slowly. Defaults to 100.
    """
    def __init__(self, warm_up=100, **kwargs):
        super(LinearDecayEMAHook, self).__init__(**kwargs)
        self.decay_fun = lambda x: 1. - min((1. - self.decay) ** self.interval,
                                        (1 + x) / (warm_up + x))
