# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

dp_factory = {'cuda': MMDataParallel, 'cpu': MMDataParallel}

ddp_factory = {'cuda': MMDistributedDataParallel}


def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is cuda, return a MMDataParallel model; if device is mlu,
    return a MLUDataParallel model.

    Args:
        model (:class:`nn.Module`): model to be parallelized.
        device (str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        nn.Module: the model to be parallelized.
    """
    if device == 'cuda':
        model = model.cuda(kwargs['device_ids'][0])
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    return dp_factory[device](model, dim=dim, *args, **kwargs)


def build_ddp(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel model;
    if device is mlu, return a MLUDistributedDataParallel model.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.

    Returns:
        :class:`nn.Module`: the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'mlu'], 'Only available for cuda or mlu devices.'
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()

    return ddp_factory[device](model, *args, **kwargs)


def is_mlu_available():
    """Returns a bool indicating if MLU is currently available."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()


def get_device():
    """Returns an available device, cpu, cuda or mlu."""
    is_device_available = {
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available()
    }
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) == 1 else 'cpu'
