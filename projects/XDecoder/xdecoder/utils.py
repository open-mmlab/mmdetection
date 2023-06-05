import logging
from contextlib import contextmanager
from functools import wraps

import torch
from torch.nn import functional as F


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


def retry_if_cuda_oom(func):
    """Makes a function retry itself after encountering pytorch's CUDA OOM
    error. It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
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

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info(
            'Attempting to copy inputs of {} to CPU due to CUDA OOM'.format(
                str(func)[0:5]))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, :img_size[0], :img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result,
        size=(output_height, output_width),
        mode='bicubic',
        align_corners=False,
        antialias=True)[0]
    return result
