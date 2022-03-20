# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py  # noqa
import warnings
from collections import abc
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch
from mmcv.runner.fp16_utils import cast_tensor_type

from mmdet.utils import get_root_logger


def cast_tensor_device(inputs, src_device, dst_device):
    """Recursively convert Tensor in inputs from src_device to dst_device.

    Args:
        inputs: Inputs that to be casted.
        src_device (torch.device): Source device..
        dst_device (torch.device): Destination device.

    Returns:
        The same device with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, torch.Tensor):
        # we need to ensure that the type of inputs to be casted are the same
        # as the argument `src_type`.
        return inputs.to(dst_device) \
            if inputs.device == src_device else inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_device(v, src_device, dst_device)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_device(item, src_device, dst_device)
            for item in inputs)
    else:
        return inputs


@contextmanager
def _ignore_torch_cuda_oom():
    """A context which ignores CUDA OOM exception from pytorch."""
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if 'CUDA out of memory. ' in str(e):
            pass
        else:
            raise


class AvoidOOM(object):
    """Try to convert inputs to FP16 and CPU if got a PyTorch's CUDA Out of
    Memory error.

    Args:
        keep_type (bool): Whether to ensure that tensors have the same type
            when they are passed in and processed out. If False, the output
            type will be fp16. Default: True.
        convert_cpu (bool): Whether to convert outputs to CPU if get an OOM
            error. This will slows down the code significantly.
            Default: True.
        return_gpu (bool): Whether convert outputs back to GPU, which will
            used in when `convert_cpu` is True. Default: True.

    Examples:
        >>> from mmdet.utils.memory import AvoidOOM
        >>> AvoidOOM = AvoidOOM()
        >>> output = AvoidOOM.retry_if_cuda_oom(
        >>>     some_torch_function)(input1, input2)

    Note:
        1. The output may be on CPU even if inputs are on GPU. Processing
            on CPU will slows down the code significantly.
        2. When converting inputs to CPU, it will only look at each argument
            and check if it has `.device` and `.to` for conversion. Nested
            structures of tensors are not supported.
        3. Since the function might be called more than once, it has to be
            stateless.
    """

    def __init__(self, keep_type=True, convert_cpu=True, return_gpu=True):
        self.logger = get_root_logger()
        self.keep_type = keep_type
        self.convert_cpu = convert_cpu
        self.return_gpu = return_gpu

    def retry_if_cuda_oom(self, func):
        """Makes a function retry itself after encountering pytorch's CUDA OOM
        error. It will do the following steps:

            1. first retry after calling `torch.cuda.empty_cache()`.
            2. If that still fails, it will then retry by trying to
              convert inputs to FP16.
            3. If that still fails trying to convert inputs to CPUs.
              In this case, it expects the function to dispatch to
              CPU implementation.

        Args:
            func: a stateless callable that takes tensor-like objects
                as arguments
        Returns:
            func: a callable which retries `func` if OOM is encountered.
        """

        def maybe_to_cpu(x):
            try:
                like_gpu_tensor = x.device.type == 'cuda' and hasattr(x, 'to')
            except AttributeError:
                like_gpu_tensor = False
            if like_gpu_tensor:
                return x.to(device='cpu')
            else:
                return x

        def maybe_to_fp16(x):
            try:
                like_float_tensor = \
                    x.dtype == torch.float32 or x.dtype == torch.float64
            except AttributeError:
                like_float_tensor = False
            if like_float_tensor:
                return x.to(torch.half)
            else:
                return x

        @wraps(func)
        def wrapped(*args, **kwargs):

            # raw function
            with _ignore_torch_cuda_oom():
                return func(*args, **kwargs)

            # Clear cache and retry
            torch.cuda.empty_cache()
            with _ignore_torch_cuda_oom():
                return func(*args, **kwargs)

            # Try to use FP16
            fp16_args = (maybe_to_fp16(x) for x in args)
            fp16_kwargs = {k: maybe_to_fp16(v) for k, v in kwargs.items()}
            self.logger.info(f'Attempting to copy inputs of {str(func)} '
                             f'to fp16 due to CUDA OOM')

            if self.keep_type:
                # get input tensor type, the output type will same as
                # the first parameter type.
                if len(args) > 0:
                    dtype = args[0].dtype
                elif len(kwargs) > 0:
                    dtype = list(kwargs.values())[0].dtype
                else:
                    ValueError('The length of inputs is 0')

                with _ignore_torch_cuda_oom():
                    self.logger.info(f'Trying to convert output to {dtype}')
                    output = func(*fp16_args, **fp16_kwargs)
                    output = cast_tensor_type(
                        output, src_type=torch.half, dst_type=dtype)
                    return output
                self.logger.info(f'Cannot convert output to {dtype} due to '
                                 'CUDA OOM.')

            with _ignore_torch_cuda_oom():
                # try to convert outputs to fp16
                self.logger.info('Trying to convert outputs to fp16')
                return func(*fp16_args, **fp16_kwargs)
            warnings.warn('Cannot convert outputs to fp16 due to CUDA OOM')
            self.logger.info('Cannot convert output to fp16 due to '
                             'CUDA OOM.')

            # Try on CPU. This slows down the code significantly,
            # therefore print a notice.
            if self.convert_cpu:
                self.logger.info(f'Attempting to copy inputs of {str(func)} '
                                 f'to CPU due to CUDA OOM')
                cpu_args = (maybe_to_cpu(x) for x in args)
                cpu_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
                if self.return_gpu:
                    # try to convert outputs to GPU
                    if len(args) > 0:
                        device = args[0].device
                    elif len(kwargs) > 0:
                        device = list(kwargs.values())[0].device
                    with _ignore_torch_cuda_oom():
                        self.logger.info(f'Convert outputs to GPU '
                                         f'(device={device})')
                        output = func(*cpu_args, **cpu_kwargs)
                        src_type = torch.zeros(0).device
                        output = cast_tensor_device(
                            output, src_device=src_type, dst_device=device)
                        return output

                warnings.warn('Cannot convert output to GPU due to CUDA OOM, '
                              'the output is now on CPU, which might cause '
                              'errors if the output need to interact with GPU '
                              'data in subsequent operations')
                self.logger.info('Cannot convert output to GPU due to '
                                 'CUDA OOM, the output is on CPU now.')

                return func(*cpu_args, **cpu_kwargs)
            else:
                # may still get CUDA OOM error
                return func(*fp16_args, **fp16_kwargs)

        return wrapped
