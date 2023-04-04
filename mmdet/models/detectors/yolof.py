# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class YOLOF(SingleStageDetector):
    r"""Implementation of `You Only Look One-level Feature
    <https://arxiv.org/abs/2103.09460>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLOF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
