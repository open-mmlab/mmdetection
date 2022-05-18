# Copyright (c) OpenMMLab. All rights reserved.
from .inference import (async_inference_detector, inference_detector,
                        init_detector)

__all__ = [
    'init_detector',
    'async_inference_detector',
    'inference_detector',
]
