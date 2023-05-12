# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import numpy as np
import torch

from mmdet.registry import MODELS
from mmdet.utils import reduce_mean
from .base import BaseDetector
from .multi_stream_detector import MultiSteamDetector

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


@MODELS.register_module()
class ConsistentTeacher(MultiSteamDetector):

    def __init__(self,
                 model: Dict[str, BaseDetector],
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg)
