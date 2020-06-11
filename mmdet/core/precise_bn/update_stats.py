# Modified from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/precise_bn.py  # noqa

import warnings

import torch
from mmcv.parallel import is_parallel_module
from torch.nn import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


@torch.no_grad()
def update_bn_stats(model, data_loader, num_iters=200):
    """
    Recompute and update the batch norm stats to make them more precise. During
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
        warnings.warn('No BN found in model')
        return

    # Finds all the other norm layers with training=True.
    for m in model.modules():
        if m.training and isinstance(m, (_InstanceNorm, GroupNorm)):
            warnings.warn('IN/GN stats will be updated like training.')

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

    for ind, data in enumerate(data_loader):
        with torch.no_grad():
            parallel_module(**data)
        for i, bn in enumerate(bn_layers):
            # Accumulates the bn stats.
            running_mean[i] += (bn.running_mean - running_mean[i]) / (i + 1)
            running_var[i] += (bn.running_var - running_var[i]) / (i + 1)
            # We compute the "average of variance" across iterations.
        if ind >= num_iters:
            break

    for i, bn in enumerate(bn_layers):
        # Sets the precise bn stats.
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.momentum = momentum_actual[i]
