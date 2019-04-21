from collections import abc

import numpy as np
import torch


# copy updated param from fp32_weight to fp16 net
def copy_in_params(fp16_net, fp32_weight):
    for net_param, fp32_weight_param in zip(fp16_net.parameters(),
                                            fp32_weight):
        net_param.data.copy_(fp32_weight_param.data)


# copy gradient from fp16 net to fp32 weight copy
def set_grad(fp16_net, fp32_weight):
    for param, param_w_grad in zip(fp32_weight, fp16_net.parameters()):
        if param_w_grad.grad is not None:
            if param.grad is None:
                param.grad = param.data.new(*param.data.size())
            param.grad.data.copy_(param_w_grad.grad.data)


def convert(inputs, src_type, dst_type, min_dim=0):
    if isinstance(inputs, torch.Tensor):
        # some tensor don't need to convert to fp16, e.g. gt_bboxes
        # these tensors' dim are usually smaller or equal to 2
        if inputs.dim() > min_dim and inputs.dtype == src_type:
            inputs = inputs.to(dst_type)
        return inputs
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: convert(v, src_type, dst_type, min_dim)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            convert(item, src_type, dst_type, min_dim) for item in inputs)
    else:
        return inputs


def patch_forward_module(old_forward):
    # conver input to fp32
    # convert output to fp16

    def new_forward(x):
        old_output = old_forward(convert(x, x.dtype, torch.float, min_dim=0))
        return convert(old_output, torch.float, x.dtype, min_dim=0)

    return new_forward


def patch_forward_models(old_forward):
    # conver input images to fp16

    def new_forward(*args, **kwargs):
        old_output = old_forward(
            *convert(args, torch.float, torch.half, min_dim=2),
            **convert(kwargs, torch.float, torch.half, min_dim=2))
        return old_output

    return new_forward


def patch_func(old_func, src_type, dst_type):
    # convert input from src_type to dst_type

    def new_func(*args, **kwargs):
        old_output = old_func(*convert(args, src_type, dst_type, min_dim=0),
                              **convert(kwargs, src_type, dst_type, min_dim=0))
        return old_output

    return new_func


def register_float_func(detector):
    # convert torch.half input to torch.float
    # e.g. class_scores, bbox_preds
    patch_funcs = ('loss', 'get_bboxes', 'get_det_bboxes', 'refine_bboxes',
                   'regress_by_class')
    for m in detector.modules():
        for func in patch_funcs:
            if hasattr(m, func):
                setattr(m, func,
                        patch_func(getattr(m, func), torch.half, torch.float))
        for child in m.children():
            register_float_func(child)


# convert batch norm layer to fp32
def bn_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
        module.forward = patch_forward_module(module.forward)
    for child in module.children():
        bn_convert_float(child)
    return module


def wrap_fp16_model(model, convert_bn):
    # convert model to fp16
    model.forward = patch_forward_models(model.forward)
    model.half()
    if convert_bn:
        bn_convert_float(model)  # bn should be in fp32
    register_float_func(model)
