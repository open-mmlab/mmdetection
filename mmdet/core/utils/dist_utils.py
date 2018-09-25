import os
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn.utils import clip_grad
from mmcv.torchpack import Hook, OptimizerHook


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    raise NotImplementedError


def _init_dist_slurm(backend, **kwargs):
    raise NotImplementedError


# modified from https://github.com/NVIDIA/apex/blob/master/apex/parallel/distributed.py#L9
def coalesce_all_reduce(tensors):
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)

    for tp in buckets:
        bucket = buckets[tp]
        coalesced = _flatten_dense_tensors(bucket)
        dist.all_reduce(coalesced)
        coalesced /= dist.get_world_size()

        for buf, synced in zip(bucket,
                               _unflatten_dense_tensors(coalesced, bucket)):
            buf.copy_(synced)


def reduce_grads(model, coalesce=True):
    grads = [
        param.grad.data for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    if coalesce:
        coalesce_all_reduce(grads)
    else:
        for tensor in grads:
            dist.all_reduce(tensor)


class DistOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        reduce_grads(runner.model, self.coalesce)
        if self.grad_clip is not None:
            clip_grad.clip_grad_norm_(
                filter(lambda p: p.requires_grad, runner.model.parameters()),
                **self.grad_clip)
        runner.optimizer.step()


class DistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        runner.data_loader.sampler.set_epoch(runner.epoch)
