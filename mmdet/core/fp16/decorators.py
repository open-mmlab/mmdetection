import functools
from inspect import getfullargspec

import torch

from .utils import cast_tensor_type


def auto_fp16(apply_to=None, out_fp32=False):
    # convert tensor from torch.float to torch.half
    # only convert the tensor specified in apply_to
    # if apply_to is None, then apply all tensors in args

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not args[0].fp16_enabled:
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            cast_args = args_info.args if apply_to is None else apply_to
            num_args = len(args)
            num_kwargs = len(kwargs)
            arg_names = args_info.args[:num_args]
            # convert the args specified in apply_to
            if num_args > 0:
                new_args = []
                for i, arg in enumerate(arg_names):
                    if arg in cast_args:
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
    # if apply_to is None, then apply all tensors in args

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if not args[0].fp16_enabled:
                return old_func(*args, **kwargs)
            args_info = getfullargspec(old_func)
            num_args = len(args)
            num_kwargs = len(kwargs)
            cast_args = args_info.args if apply_to is None else apply_to
            arg_names = args_info.args[:num_args]
            # convert the args specified in apply_to
            if num_args > 0:
                new_args = []
                for i, arg in enumerate(arg_names):
                    if arg in cast_args:
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
