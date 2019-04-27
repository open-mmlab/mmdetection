import torch
import torch.distributed as dist
from mmcv.runner import Hook

from .utils import copy_in_params, set_grad, wrap_fp16_model
from ..utils.dist_utils import DistOptimizerHook, allreduce_grads


class Fp16PrepareHook(Hook):
    """This hook initialize the necessary condition for mix precision training,
    e.g. copy master fp32 weight, convert bn layer to fp32

    Args:
        optimizer(int): Original optimizer
        distribute(bool): If use distribute training
        convert_bn(bool): If convert bn layer to fp32
    """  # noqa: W605

    def __init__(self, optimizer, distribute=True, convert_bn=True):
        self.optimizer = optimizer
        self.distribute = distribute
        self.convert_bn = convert_bn

    def before_run(self, runner):
        model = runner.model.module
        for m in model.modules():
            if hasattr(m, 'fp16_enabled'):
                m.fp16_enabled = True
        # fp32 weight copy
        param_copy = [param.data.clone() for param in model.parameters()]
        for param, net_param in zip(param_copy, model.parameters()):
            param.requires_grad = net_param.requires_grad
            if self.distribute:
                dist.broadcast(param, 0)
        # convert model to fp16
        wrap_fp16_model(model, convert_bn=self.convert_bn)
        runner.init_optimizer(self.optimizer)
        optim = getattr(torch.optim, self.optimizer['type'])
        self.optimizer.pop('type')
        runner.optimizer = optim(param_copy, **self.optimizer)


class Fp16OptimizerHook(DistOptimizerHook):
    """FP16 optimizer used for mix precision training, there are some extra
       steps compared with normal FP32 optimizer, e.g.
       1. Loss scale
       2. Copy gradient from FP16 model weight to FP32 weight copy
       3. Update FP32 weight copy parameters
       4. Copy updated parameters from FP32 weight copy to FP16 model weight
       
    Args:
        loss_scale(float): Scall factor multiplied with loss
        distribute(bool): If use distribute training
    """  # noqa: W605

    def __init__(self, grad_clip=None, loss_scale=512., distribute=True):
        super(Fp16OptimizerHook, self).__init__(grad_clip)
        self.loss_scale = loss_scale
        self.distribute = distribute

    def after_train_iter(self, runner):
        fp32_weight = runner.optimizer.param_groups[0]['params']
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        scaled_loss = runner.outputs['loss'] * self.loss_scale
        scaled_loss.backward()
        set_grad(runner.model, fp32_weight)
        if self.distribute:
            allreduce_grads(fp32_weight, self.coalesce, self.bucket_size_mb)
        for p in fp32_weight:
            if p.grad is not None:
                p.grad.div_(self.loss_scale)
        if self.grad_clip is not None:
            self.clip_grads(fp32_weight)
        runner.optimizer.step()
        copy_in_params(runner.model, fp32_weight)
