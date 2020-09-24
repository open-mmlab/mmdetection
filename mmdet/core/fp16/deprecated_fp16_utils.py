import warnings

from mmcv.runner import (Fp16OptimizerHook, auto_fp16, force_fp32,
                         wrap_fp16_model)


class DeprecatedFp16OptimizerHook(Fp16OptimizerHook):
    """A wrapper class for the FP16 optimizer hook. This class wraps
    :class:`Fp16OptimizerHook` in `mmcv.runner` and shows a warning that the
    :class:`Fp16OptimizerHook` from `mmdet.core` will be deprecated.

    Refer to :class:`Fp16OptimizerHook` in `mmcv.runner` for more details.

    Args:
        loss_scale (float): Scale factor multiplied with loss.
    """

    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing Fp16OptimizerHook from "mmdet.core" will be '
            'deprecated in the future. Please import them from "mmcv.runner" '
            'instead')


def deprecated_auto_fp16(*args, **kwargs):
    warnings.warn(
        'Importing auto_fp16 from "mmdet.core" will be '
        'deprecated in the future. Please import them from "mmcv.runner" '
        'instead')
    return auto_fp16(*args, **kwargs)


def deprecated_force_fp32(*args, **kwargs):
    warnings.warn(
        'Importing force_fp32 from "mmdet.core" will be '
        'deprecated in the future. Please import them from "mmcv.runner" '
        'instead')
    return force_fp32(*args, **kwargs)


def deprecated_wrap_fp16_model(*args, **kwargs):
    warnings.warn(
        'Importing wrap_fp16_model from "mmdet.core" will be '
        'deprecated in the future. Please import them from "mmcv.runner" '
        'instead')
    wrap_fp16_model(*args, **kwargs)
