import torch
from mmcv.runner import Hook, OptimizerHook

from .utils import patch_norm_fp32
from ..utils.dist_utils import allreduce_grads


class Fp16PrepareHook(Hook):
    """FP16 prepare hook.

    This hook initializes the necessary condition for FP16 training,
    e.g. copy master fp32 weight, convert bn layer to fp32.

    Args:
        optimizer (dict): Original optimizer.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def before_run(self, runner):
        model = runner.model.module
        # set fp16 flag
        for m in model.modules():
            if hasattr(m, 'fp16_enabled'):
                m.fp16_enabled = True
        # fp32 weight copy
        param_copy = [param.data.clone() for param in model.parameters()]
        for param, net_param in zip(param_copy, model.parameters()):
            param.requires_grad = net_param.requires_grad
        # convert model to fp16
        wrap_fp16_model(model)
        # wrap fp16 optimizer
        optimizer = self.optimizer.copy()
        runner.init_optimizer(optimizer)
        optim = getattr(torch.optim, optimizer['type'])
        optimizer.pop('type')
        runner.optimizer = optim(param_copy, **optimizer)


class Fp16OptimizerHook(OptimizerHook):
    """ FP16 optimizer hook.

    Compared with normal FP32 optimizer, there are some extra steps. e.g:
       1. Scale loss.
       2. Copy gradient from FP16 model to FP32 weight copy.
       3. Update FP32 weight copy parameters.
       4. Copy updated parameters from FP32 weight copy to FP16 model.

    Args:
        loss_scale (float): Scale factor multiplied with loss.
        distribute (bool): If use distributed training.
    """

    def __init__(self,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,
                 distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.loss_scale = loss_scale
        self.distributed = distributed

    def after_train_iter(self, runner):
        fp32_weight = runner.optimizer.param_groups[0]['params']
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        scaled_loss = runner.outputs['loss'] * self.loss_scale
        scaled_loss.backward()
        set_grad(runner.model, fp32_weight)
        if self.distributed:
            allreduce_grads(fp32_weight, self.coalesce, self.bucket_size_mb)
        for p in fp32_weight:
            if p.grad is not None:
                p.grad.div_(self.loss_scale)
        if self.grad_clip is not None:
            self.clip_grads(fp32_weight)
        runner.optimizer.step()
        copy_in_params(runner.model, fp32_weight)


# copy updated param from fp32_weight to fp16 net
def copy_in_params(fp16_net, fp32_weight):
    for net_param, fp32_weight_param in zip(fp16_net.parameters(),
                                            fp32_weight):
        net_param.data.copy_(fp32_weight_param.data)


# copy gradient from fp16 net to fp32 weight copy
def set_grad(fp16_net, fp32_weight):
    for param_fp32, param_fp16 in zip(fp32_weight, fp16_net.parameters()):
        if param_fp16.grad is not None:
            if param_fp32.grad is None:
                param_fp32.grad = param_fp32.data.new(*param_fp32.data.size())
            param_fp32.grad.data.copy_(param_fp16.grad.data)


def wrap_fp16_model(model):
    # convert model to fp16
    model.half()
    patch_norm_fp32(model)  # bn should be in fp32
