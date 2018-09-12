import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.utils import clip_grad
from mmcv.torchpack import Hook, OptimizerStepperHook

__all__ = [
    'init_dist', 'average_gradients', 'broadcast_params',
    'DistOptimizerStepperHook', 'DistSamplerSeedHook'
]


def init_dist(world_size,
              rank,
              backend='gloo',
              master_ip='127.0.0.1',
              port=29500):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(port)
    if backend == 'nccl':
        dist.init_process_group(backend='nccl')
    else:
        dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)


def average_gradients(model):
    for param in model.parameters():
        if param.requires_grad and not (param.grad is None):
            dist.all_reduce(param.grad.data)


def broadcast_params(model):
    for p in model.state_dict().values():
        dist.broadcast(p, 0)


class DistOptimizerStepperHook(OptimizerStepperHook):

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        average_gradients(runner.model)
        if self.grad_clip:
            clip_grad.clip_grad_norm_(
                filter(lambda p: p.requires_grad, runner.model.parameters()),
                max_norm=self.max_norm,
                norm_type=self.norm_type)
        runner.optimizer.step()


class DistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        runner.data_loader.sampler.set_epoch(runner.epoch)
