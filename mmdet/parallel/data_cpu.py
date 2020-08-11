# Copyright (c) Open-MMLab. All rights reserved.
import torch
from mmcv.parallel.data_container import DataContainer
from mmcv.parallel import MMDataParallel


def scatter_cpu(inputs):
    """Scatter inputs to cpu.
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return [obj]
        if isinstance(obj, DataContainer):
            return obj.data
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter_cpu(inputs) if inputs else []
    kwargs = scatter_cpu(kwargs) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class MMDataCPU(MMDataParallel):
    """Implementation of MMDataParallel to use CPU for training"""

    def scatter(self, inputs, kwargs):
        return scatter_kwargs(inputs, kwargs)

    def train_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        return self.module.val_step(*inputs[0], **kwargs[0])

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs)
        return self.module(*inputs[0], **kwargs[0])


