import torch
import torch.nn as nn
from mmcv.runner import Hook, OptimizerHook

from .utils import cast_tensor_type
from ..utils.dist_utils import allreduce_grads


class Fp16PrepareHook(Hook):
    """FP16 preparation hook.

    This hook initializes the necessary condition for FP16 training,
    e.g. copy master fp32 weight, convert bn layer to fp32.

    Args:
        optimizer_cfg (dict): Original optimizer config.
    """

    def __init__(self, optimizer_cfg):
        self.optimizer_cfg = optimizer_cfg

    def before_run(self, runner):
        model = runner.model.module
        # keep a copy of fp32 weights
        param_copy = []
        for param in model.parameters():
            copied = param.data.clone()
            copied.requires_grad = param.requires_grad
            param_copy.append(copied)
        # convert model to fp16
        wrap_fp16_model(model)
        # use the fp32 weights to build the optimizer
        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_type = optimizer_cfg.pop('type')
        optimizer_cls = getattr(torch.optim, optimizer_type)
        runner.optimizer = optimizer_cls(param_copy, **optimizer_cfg)


class Fp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook.

    The steps of fp16 optimizer is as follows.
    1. Scale the loss value.
    2. BP in the fp16 model.
    2. Copy gradients from fp16 model to fp32 weights.
    3. Update fp32 weights.
    4. Copy updated parameters from fp32 weights to fp16 model.

    Refer to https://arxiv.org/abs/1710.03740 for more details.

    Args:
        loss_scale (float): Scale factor multiplied with loss.
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

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(*fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        # scale the loss value
        scaled_loss = runner.outputs['loss'] * self.loss_scale
        scaled_loss.backward()
        # copy fp16 grads in the model to fp32 params in the optimizer
        fp32_weights = runner.optimizer.param_groups[0]['params']
        self.copy_grads_to_fp32(runner.model, fp32_weights)
        # allreduce grads
        if self.distributed:
            allreduce_grads(fp32_weights, self.coalesce, self.bucket_size_mb)
        # scale the gradients back
        for param in fp32_weights:
            if param.grad is not None:
                param.grad.div_(self.loss_scale)
        if self.grad_clip is not None:
            self.clip_grads(fp32_weights)
        # update fp32 params
        runner.optimizer.step()
        # copy fp32 params to the fp16 model
        self.copy_params_to_fp16(runner.model, fp32_weights)


def wrap_fp16_model(model):
    # convert model to fp16
    model.half()
    # patch the normalization layers to make it work in fp32 mode
    patch_norm_fp32(model)
    # set `fp16_enabled` flag
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True


def patch_norm_fp32(module):
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
        module.float()
        module = patch_forward_method(module, torch.half, torch.float)
    for child in module.children():
        patch_norm_fp32(child)
    return module


def patch_forward_method(module, src_type, dst_type, convert_output=True):
    """Patch the forward method of a module.

    Args:
        module (:obj:`nn.Module`): The module to be patched.
        src_type (torch.dtype): Type of input arguments to be converted from.
        dst_type (torch.dtype): Type of input arguments to be converted to.
        convert_output (bool): Whether to convert the output back to src_type.

    Returns:
        nn.Module: The patched module.
    """

    def new_forward(*args, **kwargs):
        output = module.forward(*cast_tensor_type(args, src_type, dst_type),
                                **cast_tensor_type(kwargs, src_type, dst_type))
        if convert_output:
            output = cast_tensor_type(output, dst_type, src_type)
        return output

    module.forward = new_forward

    return module
