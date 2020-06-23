# Copyright (c) 2019 Western Digital Corporation or its affiliates.

from .single_stage import SingleStageDetector
from ..builder import DETECTORS


@DETECTORS.register_module
class YoloNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YoloNet, self).__init__(backbone,
                                      neck,
                                      bbox_head,
                                      train_cfg,
                                      test_cfg,
                                      pretrained)
