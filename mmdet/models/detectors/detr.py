# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .detection_transformer import TransformerDetector


@DETECTORS.register_module()
class DETR(TransformerDetector):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)
