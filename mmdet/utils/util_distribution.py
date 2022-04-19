# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

dp_factory = {
        'cuda' : MMDataParallel,
        'cpu' : MMDataParallel
    }

ddp_factory = {
        'cuda' : MMDistributedDataParallel
    }

def build_dp(model, device='cuda', device_ids=None):
    """build DataParallel module by device type.
    
    if device is cuda, return a MMDataParallel model; if device is mlu,
    return a MLUDataParallel model.

    Args:
        model(nn.Moudle): model to be parallelized.
        device(str): device type, cuda, cpu or mlu. Defaults to cuda.
        device_ids(int): device ids of modules to be scattered to.
            Defaults to None when GPU or MLU is not available.

    Returns:
        model(nn.Module): the model to be parallelized.
    """
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    return dp_factory[device](model, device_ids=device_ids)


def build_ddp(model, device='cuda', device_ids=None,
        broadcast_buffers=False, find_unused_parameters=False):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel model; if device is mlu,
    return a MLUDistributedDataParallel model.

    Args:
        model(:class:`nn.Moudle`): module to be parallelized.
        device(str): device type, mlu or cuda.
        device_ids(int): which represents the only device where the input module
            corresponding to this process resides. Defaults to None.
        broadcast_buffers(bool): Flag that enables syncing (broadcasting) buffers of
            the module at beginning of the forward function. Defaults to True.
        find_unused_parameters(bool): Traverse the autograd graph of all tensors
            contained in the return value of the wrapped module's ``forward`` function.
            Parameters that don't receive gradients as part of this graph are preemptively
            marked as being ready to be reduced. Note that all ``forward`` outputs that
            are derived from module parameters must participate in calculating loss and
            later the gradient computation. If they don't, this wrapper will hang waiting
            for autograd to produce gradients for those parameters. Any outputs derived from
            module parameters that are otherwise unused can be detached from the autograd
            graph using ``torch.Tensor.detach``. Defaults to False.

    Returns:
        model(nn.Module): the module to be parallelized

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'mlu'], 'Only available for cuda or mlu devices.'
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import  MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()

    return ddp_factory[device](model, device_ids=device_ids,
                               broadcast_buffers=broadcast_buffers,
                               find_unused_parameters=find_unused_parameters)

def is_mlu_available():
    """ Returns a bool indicating if MLU is currently available. """
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()

def get_device():
    """ Returns an available device, cpu, cuda or mlu. """
    is_device_available = {'cuda': torch.cuda.is_available(),
                           'mlu': is_mlu_available()}
    device_list = [k for k, v in is_device_available.items() if v ]
    return device_list[0] if len(device_list) == 1 else 'cpu'
