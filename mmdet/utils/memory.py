# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py  # noqa
import warnings
from contextlib import contextmanager
from functools import wraps

import torch

from mmdet.utils import get_root_logger


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
        logger (logging.Logger | str, optional): Logger used for printing
            related information during evaluation. Default: None.
        return_gpu (bool): Whether convert outputs back to GPU. Default: True.
        out_fp32 (bool): Whether to convert the output back to out_fp32.
            Default: True.
        output_type (torch.dtype): Destination type. Default: None

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
                 logger=None,
                 out_fp32=True,
                 return_gpu=True,
                 output_type=None,
                 convert_cpu=True):
        self.logger = logger
        self.return_gpu = return_gpu
        if output_type is not None:
            self.out_fp32 = True
        else:
            self.out_fp32 = out_fp32
        self.output_type = output_type
        self.convert_cpu = convert_cpu

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
            if self.logger is None:
                self.logger = get_root_logger()

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
            if self.out_fp32:
                if self.output_type is not None:
                    dtype = self.output_type
                else:
                    dtype = torch.float
                with _ignore_torch_cuda_oom():
                    self.logger.info(f'Trying to convert output to {dtype}')
                    return func(*fp16_args, **fp16_kwargs).to(dtype=dtype)
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
                        return func(*cpu_args, **cpu_kwargs).to(device=device)

                warnings.warn('Cannot convert output to GPU due to CUDA OOM, '
                              'the output is now on CPU, which might cause '
                              'errors if the output need to interact with GPU '
                              'data in subsequent operations')
                self.logger.info('Cannot convert output to GPU due to '
                                 'CUDA OOM, the output is on CPU now.')

                return func(*cpu_args, **cpu_kwargs)
            else:
                return func(*fp16_args, **fp16_kwargs)

        return wrapped
