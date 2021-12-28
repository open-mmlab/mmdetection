# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class GridRCNN(TwoStageDetector):
    """Grid R-CNN.

    This detector is the implementation of:
    - Grid R-CNN (https://arxiv.org/abs/1811.12030)
    - Grid R-CNN Plus: Faster and Better (https://arxiv.org/abs/1906.05688)
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(GridRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
