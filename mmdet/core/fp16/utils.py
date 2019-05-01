import functools
from collections import abc
from inspect import getfullargspec

import numpy as np
import torch
import torch.nn as nn


def cast_tensor_type(inputs, src_type, dst_type, min_dim=0):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type, min_dim)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(cast_tensor_type(item, src_type, dst_type, min_dim)
                            for item in inputs)
    else:
        return inputs


def patch_forward_module(old_forward, src_type, dst_type, convert_output):
    # conver input from src_type to dst_type

    def new_forward(*args, **kwargs):
        output = old_forward(
            *cast_tensor_type(args, src_type, dst_type, min_dim=0),
            **cast_tensor_type(kwargs, dst_type, src_type, min_dim=0))
        if convert_output:
            output = cast_tensor_type(output, dst_type, src_type, min_dim=0)
        return output

    return new_forward


def patch_norm_fp32(module):
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
        module.float()
        module.forward = patch_forward_module(
            module.forward, torch.half, torch.float, convert_output=True)
    for child in module.children():
        patch_norm_fp32(child)
    return module


def auto_fp16(apply_to=None, out_fp32=False):
    # convert tensor from torch.float to torch.half
    # only convert the tensor specified in apply_to

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not args[0].fp16_enabled:
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            num_args = len(args)
            num_kwargs = len(kwargs)
            arg_names = args_info.args[:num_args]
            # convert the args specified in apply_to
            if num_args > 0:
                new_args = []
                for i, arg in enumerate(arg_names):
                    if arg in apply_to:
                        new_args.append(
                            cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            else:
                new_args = args
            # convert the kwargs specified in apply_to
            if num_kwargs > 0:
                new_kwargs = dict()
                for k, v in kwargs.items():
                    if k in apply_to:
                        new_kwargs[k] = cast_tensor_type(
                            v, torch.float, torch.half)
                    else:
                        new_kwargs[k] = v
            else:
                new_kwargs = kwargs
            output = old_func(*new_args, **new_kwargs)
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper


def force_fp32(apply_to=None, out_fp16=False):
    # convert tensor from torch.half to torch.float
    # only convert the tensor specified in apply_to

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not args[0].fp16_enabled:
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            num_args = len(args)
            num_kwargs = len(kwargs)
            arg_names = args_info.args[:num_args]
            # convert the args specified in apply_to
            if num_args > 0:
                new_args = []
                for i, arg in enumerate(arg_names):
                    if arg in apply_to:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            else:
                new_args = args
            # convert the kwargs specified in apply_to
            if num_kwargs > 0:
                new_kwargs = dict()
                for k, v in kwargs.items():
                    if k in apply_to:
                        new_kwargs[k] = cast_tensor_type(
                            v, torch.half, torch.float)
                    else:
                        new_kwargs[k] = v
            else:
                new_kwargs = kwargs
            output = old_func(*new_args, **new_kwargs)
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper
