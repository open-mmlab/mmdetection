# Adapted from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/precise_bn.py  # noqa: E501
# Original licence: Copyright (c) 2019 Facebook, Inc under the Apache License 2.0  # noqa: E501

import logging
import time

import mmcv
import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import Hook
from mmcv.utils import print_log
from torch.nn import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader


def is_parallel_module(module):
    """Check if a module is a parallel module.
    The following 3 modules (and their subclasses) are regarded as parallel
    modules: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version).
    Args:
        module (nn.Module): The module to be checked.
    Returns:
        bool: True if the input module is a parallel module.
    """
    parallels = (DataParallel, DistributedDataParallel,
                 MMDistributedDataParallel)
    return bool(isinstance(module, parallels))


@torch.no_grad()
def update_bn_stats(model, data_loader, num_iters=200, logger=None):
    """Recompute and update the batch norm stats to make them more precise.
    During
    training both BN stats and the weight are changing after every iteration,
    so the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true
    average of per-batch mean/variance instead of the running average.
    Args:
        model (nn.Module): The model whose bn stats will be recomputed.
        data_loader (iterator): The DataLoader iterator.
        num_iters (int): number of iterations to compute the stats.
        logger (:obj:`logging.Logger` | None): Logger for logging.
            Default: None.
    """

    model.train()

    assert len(data_loader) >= num_iters, (
        f'length of dataloader {len(data_loader)} must be greater than '
        f'iteration number {num_iters}')

    if is_parallel_module(model):
        parallel_module = model
        model = model.module
    else:
        parallel_module = model
    # Finds all the bn layers with training=True.
    bn_layers = [
        m for m in model.modules() if m.training and isinstance(m, _BatchNorm)
    ]

    if len(bn_layers) == 0:
        print_log('No BN found in model', logger=logger, level=logging.WARNING)
        return
    print_log(f'{len(bn_layers)} BN found', logger=logger)

    # Finds all the other norm layers with training=True.
    for m in model.modules():
        if m.training and isinstance(m, (_InstanceNorm, GroupNorm)):
            print_log(
                'IN/GN stats will be updated like training.',
                logger=logger,
                level=logging.WARNING)

    # In order to make the running stats only reflect the current batch, the
    # momentum is disabled.
    # bn.running_mean = (1 - momentum) * bn.running_mean + momentum *
    # batch_mean
    # Setting the momentum to 1.0 to compute the stats without momentum.
    momentum_actual = [bn.momentum for bn in bn_layers]  # pyre-ignore
    for bn in bn_layers:
        bn.momentum = 1.0

    # Note that running_var actually means "running average of variance"
    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    finish_before_loader = False
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for ind, data in enumerate(data_loader):
        with torch.no_grad():
            parallel_module(**data, return_loss=False)
        prog_bar.update()
        for i, bn in enumerate(bn_layers):
            # Accumulates the bn stats.
            running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
            # running var is actually
            running_var[i] += (bn.running_var - running_var[i]) / (ind + 1)

        if (ind + 1) >= num_iters:
            finish_before_loader = True
            break
    assert finish_before_loader, 'Dataloader stopped before ' \
                                 f'iteration {num_iters}'

    for i, bn in enumerate(bn_layers):
        # Sets the precise bn stats.
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.momentum = momentum_actual[i]


class PreciseBNHook(Hook):
    """Precise BN hook.
    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        num_iters (int): Number of iterations to update the bn stats.
            Default: 200.
        interval (int): Perform precise bn interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, num_iters=200, interval=1):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.num_iters = num_iters

    def after_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.interval):
            # sleep to avoid possible deadlock
            time.sleep(2.)
            print_log(
                f'Running Precise BN for {self.num_iters} iterations',
                logger=runner.logger)
            update_bn_stats(
                runner.model,
                self.dataloader,
                self.num_iters,
                logger=runner.logger)
            print_log('BN stats updated', logger=runner.logger)
            # sleep to avoid possible deadlock
            time.sleep(2.)
