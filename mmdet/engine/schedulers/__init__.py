# Copyright (c) OpenMMLab. All rights reserved.
from .quadratic_warmup import (QuadraticWarmupLR, QuadraticWarmupMomentum,
                               QuadraticWarmupParamScheduler)

__all__ = [
    'QuadraticWarmupParamScheduler', 'QuadraticWarmupMomentum',
    'QuadraticWarmupLR'
]
