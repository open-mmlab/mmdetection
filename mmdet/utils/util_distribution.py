# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

dp_factory = {'cuda': MMDataParallel, 'cpu': MMDataParallel}

ddp_factory = {'cuda': MMDistributedDataParallel}


def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """按设备类型构建 DataParallel 模块.

    如果设备是 cuda，则返回一个 MMDistributedDataParallel 模型;
    如果设备是 mlu，则返回一个 MLUDistributedDataParallel 模型.

    Args:
        model (:class:`nn.Module`): 要并行化的模型.
        device (str): 设备类型, cuda, cpu or mlu. 默认为 cuda.
        dim (int): 用于分散数据的维度.默认为 0

    Returns:
        nn.Module: 并行化的模型.
    """
    if device == 'cuda':
        model = model.cuda(kwargs['device_ids'][0])
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    return dp_factory[device](model, dim=dim, *args, **kwargs)


def build_ddp(model, device='cuda', *args, **kwargs):
    """按设备类型构建 DistributedDataParallel 模块.

    如果设备是 cuda，则返回一个 MMDistributedDataParallel 模型;
    如果设备是 mlu，则返回一个 MLUDistributedDataParallel 模型.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.

    Returns:
        :class:`nn.Module`: the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'mlu'], '仅适用于 cuda 或 mlu 设备.'
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
