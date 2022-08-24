# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import abc
from contextlib import contextmanager
from functools import wraps

import torch
from mmengine.logging import MMLogger


def cast_tensor_type(inputs, src_type=None, dst_type=None):
    """Recursively convert Tensor in inputs from ``src_type`` to ``dst_type``.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype | torch.device): Source type.
        src_type (torch.dtype | torch.device): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    assert dst_type is not None
    if isinstance(inputs, torch.Tensor):
        if isinstance(dst_type, torch.device):
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
            k: cast_tensor_type(v, src_type=src_type, dst_type=dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type=src_type, dst_type=dst_type)
            for item in inputs)
    # TODO: Currently not supported
    # elif isinstance(inputs, InstanceData):
    #     for key, value in inputs.items():
    #         inputs[key] = cast_tensor_type(
    #             value, src_type=src_type, dst_type=dst_type)
    #     return inputs
    else:
        return inputs


@contextmanager
def _ignore_torch_cuda_oom():
    """A context which ignores CUDA OOM exception from pytorch.

    Code is modified from
    <https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py>  # noqa: E501
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if 'CUDA out of memory. ' in str(e):
            pass
        else:
            raise


class AvoidOOM:
    """Try to convert inputs to FP16 and CPU if got a PyTorch's CUDA Out of
    Memory error. It will do the following steps:

        1. First retry after calling `torch.cuda.empty_cache()`.
        2. If that still fails, it will then retry by converting inputs
          to FP16.
        3. If that still fails trying to convert inputs to CPUs.
          In this case, it expects the function to dispatch to
          CPU implementation.

    Args:
        to_cpu (bool): Whether to convert outputs to CPU if get an OOM
            error. This will slow down the code significantly.
            Defaults to True.
        test (bool): Skip `_ignore_torch_cuda_oom` operate that can use
            lightweight data in unit test, only used in
            test unit. Defaults to False.

    Examples:
        >>> from mmdet.utils.memory import AvoidOOM
        >>> AvoidCUDAOOM = AvoidOOM()
        >>> output = AvoidOOM.retry_if_cuda_oom(
        >>>     some_torch_function)(input1, input2)
        >>> # To use as a decorator
        >>> # from mmdet.utils import AvoidCUDAOOM
        >>> @AvoidCUDAOOM.retry_if_cuda_oom
        >>> def function(*args, **kwargs):
        >>>     return None
    ```

    Note:
        1. The output may be on CPU even if inputs are on GPU. Processing
            on CPU will slow down the code significantly.
        2. When converting inputs to CPU, it will only look at each argument
            and check if it has `.device` and `.to` for conversion. Nested
            structures of tensors are not supported.
        3. Since the function might be called more than once, it has to be
            stateless.
    """

    def __init__(self, to_cpu=True, test=False):
        self.to_cpu = to_cpu
        self.test = test

    def retry_if_cuda_oom(self, func):
        """Makes a function retry itself after encountering pytorch's CUDA OOM
        error.

        The implementation logic is referred to
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py

        Args:
            func: a stateless callable that takes tensor-like objects
                as arguments.
        Returns:
            func: a callable which retries `func` if OOM is encountered.
        """  # noqa: W605

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

            # Convert to FP16
            fp16_args = cast_tensor_type(args, dst_type=torch.half)
            fp16_kwargs = cast_tensor_type(kwargs, dst_type=torch.half)
            logger = MMLogger.get_current_instance()
            logger.warning(f'Attempting to copy inputs of {str(func)} '
                           'to FP16 due to CUDA OOM')

            # get input tensor type, the output type will same as
            # the first parameter type.
            with _ignore_torch_cuda_oom():
                output = func(*fp16_args, **fp16_kwargs)
                output = cast_tensor_type(
                    output, src_type=torch.half, dst_type=dtype)
                if not self.test:
                    return output
            logger.warning('Using FP16 still meet CUDA OOM')

            # Try on CPU. This will slow down the code significantly,
            # therefore print a notice.
            if self.to_cpu:
                logger.warning(f'Attempting to copy inputs of {str(func)} '
                               'to CPU due to CUDA OOM')
                cpu_device = torch.empty(0).device
                cpu_args = cast_tensor_type(args, dst_type=cpu_device)
                cpu_kwargs = cast_tensor_type(kwargs, dst_type=cpu_device)

                # convert outputs to GPU
                with _ignore_torch_cuda_oom():
                    logger.warning(f'Convert outputs to GPU (device={device})')
                    output = func(*cpu_args, **cpu_kwargs)
                    output = cast_tensor_type(
                        output, src_type=cpu_device, dst_type=device)
                    return output

                warnings.warn('Cannot convert output to GPU due to CUDA OOM, '
                              'the output is now on CPU, which might cause '
                              'errors if the output need to interact with GPU '
                              'data in subsequent operations')
                logger.warning('Cannot convert output to GPU due to '
                               'CUDA OOM, the output is on CPU now.')

                return func(*cpu_args, **cpu_kwargs)
            else:
                # may still get CUDA OOM error
                return func(*args, **kwargs)

        return wrapped


# To use AvoidOOM as a decorator
AvoidCUDAOOM = AvoidOOM()
