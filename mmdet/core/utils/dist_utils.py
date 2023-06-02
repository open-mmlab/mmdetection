# Copyright (c) OpenMMLab. All rights reserved.
import functools
import pickle
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import OptimizerHook, get_dist_info
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce gradients.

    Args:
        params (list[torch.Parameters]): List of parameters of a model
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


class DistOptimizerHook(OptimizerHook):
    """Deprecated optimizer hook for distributed training."""

    def __init__(self, *args, **kwargs):
        warnings.warn('"DistOptimizerHook" is deprecated, please switch to'
                      '"mmcv.runner.OptimizerHook".')
        super().__init__(*args, **kwargs)


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def obj2tensor(pyobj, device='cuda'):
    """Serialize picklable python object to tensor."""
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)


def tensor2obj(tensor):
    """Deserialize tensor to picklable python object."""
    return pickle.loads(tensor.cpu().numpy().tobytes())


@functools.lru_cache()
def _get_global_gloo_group():
    """Return a process group based on gloo backend, containing all the ranks
    The result is cached."""
    if dist.get_backend() == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD


def all_reduce_dict(py_dict, op='sum', group=None, to_float=True):
    """Apply all reduce function for python dict object.

    The code is modified from https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/utils/allreduce_norm.py.

    NOTE: make sure that py_dict in different ranks has the same keys and
    the values should be in the same shape. Currently only supports
    nccl backend.

    Args:
        py_dict (dict): Dict to be applied all reduce op.
        op (str): Operator, could be 'sum' or 'mean'. Default: 'sum'
        group (:obj:`torch.distributed.group`, optional): Distributed group,
            Default: None.
        to_float (bool): Whether to convert all values of dict to float.
            Default: True.

    Returns:
        OrderedDict: reduced python dict object.
    """
    warnings.warn(
        'group` is deprecated. Currently only supports NCCL backend.')
    _, world_size = get_dist_info()
    if world_size == 1:
        return py_dict

    # all reduce logic across different devices.
    py_key = list(py_dict.keys())
    if not isinstance(py_dict, OrderedDict):
        py_key_tensor = obj2tensor(py_key)
        dist.broadcast(py_key_tensor, src=0)
        py_key = tensor2obj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    if to_float:
        warnings.warn('Note: the "to_float" is True, you need to '
                      'ensure that the behavior is reasonable.')
        flatten_tensor = torch.cat(
            [py_dict[k].flatten().float() for k in py_key])
    else:
        flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_key])

    dist.all_reduce(flatten_tensor, op=dist.ReduceOp.SUM)
    if op == 'mean':
        flatten_tensor /= world_size

    split_tensors = [
        x.reshape(shape) for x, shape in zip(
            torch.split(flatten_tensor, tensor_numels), tensor_shapes)
    ]
    out_dict = {k: v for k, v in zip(py_key, split_tensors)}
    if isinstance(py_dict, OrderedDict):
        out_dict = OrderedDict(out_dict)
    return out_dict


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.

    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.

    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()
