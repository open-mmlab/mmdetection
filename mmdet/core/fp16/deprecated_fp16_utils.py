import warnings

from mmcv.runner import (Fp16OptimizerHook, auto_fp16, force_fp32,
                         wrap_fp16_model)


class Depr_Fp16OptimizerHook(Fp16OptimizerHook):

    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            'Importing Fp16OptimizerHook from "mmdet.core" will be '
            'deprecated in the future. Please import them from "mmcv.runner" '
            'instead')


def depr_auto_fp16(*args, **kwargs):
    warnings.warn(
        'Importing auto_fp16 from "mmdet.core" will be '
        'deprecated in the future. Please import them from "mmcv.runner" '
        'instead')
    return auto_fp16(*args, **kwargs)


def depr_force_fp32(*args, **kwargs):
    warnings.warn(
        'Importing force_fp32 from "mmdet.core" will be '
        'deprecated in the future. Please import them from "mmcv.runner" '
        'instead')
    return force_fp32(*args, **kwargs)


def depr_wrap_fp16_model(*args, **kwargs):
    warnings.warn(
        'Importing wrap_fp16_model from "mmdet.core" will be '
        'deprecated in the future. Please import them from "mmcv.runner" '
        'instead')
    wrap_fp16_model(*args, **kwargs)
