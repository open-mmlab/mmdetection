# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py  # noqa
import warnings
from collections import abc
from contextlib import contextmanager
from functools import wraps

import torch

from mmdet.utils import get_root_logger


def convert_tensor_type(inputs, src_type=None, dst_type=None):
    """Recursively convert Tensor in inputs from src_device to dst_device.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype or torch.device): Source type.
        dst_type (torch.dtype or torch.device): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    assert dst_type is not None
    if isinstance(inputs, torch.Tensor):
        if hasattr(dst_type, 'type'):
            # convert Tensor to dst_device
            if hasattr(inputs, 'to') and \
                    hasattr(inputs, 'device') and \
                    (inputs.device == src_type or src_type is None):
                return inputs.to(dst_type)
            else:
                return inputs
        else:
            # convert Tensor to dst_dtype
            if hasattr(inputs, 'to') and \
                    hasattr(inputs, 'dtype') and \
                    (inputs.dtype == src_type or src_type is None):
                return inputs.to(dst_type)
            else:
                return inputs
        # we need to ensure that the type of inputs to be casted are the same
        # as the argument `src_type`.
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: convert_tensor_type(v, src_type=src_type, dst_type=dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            convert_tensor_type(item, src_type=src_type, dst_type=dst_type)
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
        test (bool): Whether skip torch.cuda.empty_cache(), only used in
            test unit. Default: False
        return_fp16 (bool): Whether return torch.half, only used in test unit.
            Default: True

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

    def __init__(self,
                 keep_type=True,
                 convert_cpu=True,
                 return_gpu=True,
                 test=False,
                 return_fp16=True):
        self.logger = get_root_logger()
        self.keep_type = keep_type
        self.convert_cpu = convert_cpu
        self.return_gpu = return_gpu
        self.test = test
        self.return_fp16 = return_fp16

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

        @wraps(func)
        def wrapped(*args, **kwargs):

            # raw function
            if not self.test:
                with _ignore_torch_cuda_oom():
                    return func(*args, **kwargs)

                # Clear cache and retry
                torch.cuda.empty_cache()
                with _ignore_torch_cuda_oom():
                    return func(*args, **kwargs)

            # get the type and device of first tensor
            dtype, device = None, None
            values = args + tuple(kwargs.values())
            for value in values:
                if isinstance(value, torch.Tensor):
                    dtype = value.dtype
                    device = value.device
                    break
            if dtype is None or device is None:
                raise ValueError('There is no tensor in the inputs, '
                                 'cannot get dtype and device.')

            # Try to use FP16
            fp16_args = convert_tensor_type(args, dst_type=torch.half)
            fp16_kwargs = convert_tensor_type(kwargs, dst_type=torch.half)
            self.logger.info(f'Attempting to copy inputs of {str(func)} '
                             f'to fp16 due to CUDA OOM')

            if self.keep_type:
                # get input tensor type, the output type will same as
                # the first parameter type.
                with _ignore_torch_cuda_oom():
                    self.logger.info(f'Trying to convert output to {dtype}')
                    output = func(*fp16_args, **fp16_kwargs)
                    output = convert_tensor_type(
                        output, src_type=torch.half, dst_type=dtype)
                    return output
                self.logger.info(f'Cannot convert output to {dtype} due to '
                                 'CUDA OOM.')

            if self.return_fp16:
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
                cpu_device = torch.empty(0).device
                cpu_args = convert_tensor_type(args, dst_type=cpu_device)
                cpu_kwargs = convert_tensor_type(kwargs, dst_type=cpu_device)
                if self.return_gpu:
                    # try to convert outputs to GPU
                    with _ignore_torch_cuda_oom():
                        self.logger.info(f'Convert outputs to GPU '
                                         f'(device={device})')
                        output = func(*cpu_args, **cpu_kwargs)
                        output = convert_tensor_type(
                            output, src_type=cpu_device, dst_type=device)
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
