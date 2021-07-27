from mmcv.runner.hooks import HOOKS, Hook
import random
from mmcv.runner import get_dist_info
from collections import OrderedDict

import torch
from torch import distributed as dist
from torch import nn
import functools
import pickle


def random_resize(random_size, data_loader, rank, is_distributed, input_size):
    tensor = torch.LongTensor(2).cuda()

    if rank == 0:
        size_factor = input_size[1] * 1. / input_size[0]
        size = random.randint(*random_size)
        size = (int(32 * size), 32 * int(size * size_factor))
        tensor[0] = size[0]
        tensor[1] = size[1]

    if is_distributed:
        dist.barrier()
        dist.broadcast(tensor, 0)

    data_loader.dataset.dynamic_scale = (tensor[0].item(), tensor[1].item())
    return data_loader.dataset.dynamic_scale


ASYNC_NORM = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def get_async_norm_states(module):
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, ASYNC_NORM):
            for k, v in child.state_dict().items():
                async_norm_states[".".join([name, k])] = v
    return async_norm_states


def pyobj2tensor(pyobj, device="cuda"):
    """serialize picklable python object to tensor"""
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)


def tensor2pyobj(tensor):
    """deserialize tensor to picklable python object"""
    return pickle.loads(tensor.cpu().numpy().tobytes())


def _get_reduce_op(op_name):
    return {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
    }[op_name.lower()]


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_reduce(py_dict, op="sum", group=None):
    """
    Apply all reduce function for python dict object.
    NOTE: make sure that every py_dict has the same keys and values are in the same shape.

    Args:
        py_dict (dict): dict to apply all reduce op.
        op (str): operator, could be "sum" or "mean".
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return py_dict
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return py_dict

    # all reduce logic across different devices.
    py_key = list(py_dict.keys())
    py_key_tensor = pyobj2tensor(py_key)
    dist.broadcast(py_key_tensor, src=0)
    py_key = tensor2pyobj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    flatten_tensor = torch.cat([py_dict[k].flatten().float() for k in py_key])
    dist.all_reduce(flatten_tensor, op=_get_reduce_op(op))
    if op == "mean":
        flatten_tensor /= world_size

    split_tensors = [
        x.reshape(shape) for x, shape in zip(
            torch.split(flatten_tensor, tensor_numels), tensor_shapes
        )
    ]
    return OrderedDict({k: v for k, v in zip(py_key, split_tensors)})


def all_reduce_norm(module):
    """
    All reduce norm statistics in different devices.
    """
    states = get_async_norm_states(module)
    states = all_reduce(states, op="mean")
    module.load_state_dict(states, strict=False)


@HOOKS.register_module()
class YoloXProcessHook(Hook):
    def __init__(self, random_size=(14, 26), input_size=(640, 640), no_aug_epoch=15, eval_interval=1, change_size_interval=10):
        self.rank, world_size = get_dist_info()
        self.is_distributed = world_size > 1
        self.random_size = random_size
        self.input_size = input_size
        self.no_aug_epoch = no_aug_epoch
        self.eval_interval = eval_interval
        self.change_size_interval = change_size_interval

    def after_train_iter(self, runner):
        progress_in_iter = runner.iter
        train_loader = runner.data_loader
        # random resizing
        if self.random_size is not None and (progress_in_iter + 1) % self.change_size_interval == 0:
            random_resize(self.random_size, train_loader, self.rank, self.is_distributed, self.input_size)

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model.module
        if epoch + 1 == runner.max_epochs - self.no_aug_epoch:
            print("--->No mosaic and mixup aug now!")
            train_loader.dataset.enable_mosaic = False
            train_loader.dataset.enable_mixup = False
            print("--->Add additional L1 loss now!")
            model.bbox_head.use_l1 = True

    def after_train_epoch(self, runner):
        epoch = runner.epoch
        if (epoch + 1) % self.eval_interval == 0:
            all_reduce_norm(runner.model)
