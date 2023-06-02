# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin
from mmengine.optim.scheduler.momentum_scheduler import MomentumSchedulerMixin
from mmengine.optim.scheduler.param_scheduler import INF, _ParamScheduler
from torch.optim import Optimizer

from mmdet.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class QuadraticWarmupParamScheduler(_ParamScheduler):
    r"""Warm up the parameter value of each parameter group by quadratic
    formula:

    .. math::

        X_{t} = X_{t-1} + \frac{2t+1}{{(end-begin)}^{2}} \times X_{base}

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        param_name (str): Name of the parameter to be adjusted, such as
            ``lr``, ``momentum``.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 param_name: str,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        if end >= INF:
            raise ValueError('``end`` must be less than infinity,'
                             'Please set ``end`` parameter of '
                             '``QuadraticWarmupScheduler`` as the '
                             'number of warmup end.')
        self.total_iters = end - begin
        super().__init__(
            optimizer=optimizer,
            param_name=param_name,
            begin=begin,
            end=end,
            last_step=last_step,
            by_epoch=by_epoch,
            verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = begin * epoch_length
        if end != INF:
            end = end * epoch_length
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        if self.last_step == 0:
            return [
                base_value * (2 * self.last_step + 1) / self.total_iters**2
                for base_value in self.base_values
            ]

        return [
            group[self.param_name] + base_value *
            (2 * self.last_step + 1) / self.total_iters**2
            for base_value, group in zip(self.base_values,
                                         self.optimizer.param_groups)
        ]


@PARAM_SCHEDULERS.register_module()
class QuadraticWarmupLR(LRSchedulerMixin, QuadraticWarmupParamScheduler):
    """Warm up the learning rate of each parameter group by quadratic formula.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """


@PARAM_SCHEDULERS.register_module()
class QuadraticWarmupMomentum(MomentumSchedulerMixin,
                              QuadraticWarmupParamScheduler):
    """Warm up the momentum value of each parameter group by quadratic formula.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        begin (int): Step at which to start updating the parameters.
            Defaults to 0.
        end (int): Step at which to stop updating the parameters.
            Defaults to INF.
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.
    """
