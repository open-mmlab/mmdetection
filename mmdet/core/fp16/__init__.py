import warnings

from mmcv.runner import (Fp16OptimizerHook, auto_fp16, force_fp32,
                         wrap_fp16_model)

warnings.warn(
    'Importing from "mmdet.core.fp16" will be deprecated in'
    ' the future. Please import them from "mmcv.runner" instead', UserWarning)

__all__ = ['auto_fp16', 'force_fp32', 'Fp16OptimizerHook', 'wrap_fp16_model']
