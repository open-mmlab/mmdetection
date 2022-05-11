# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .single_stage import SingleStageDetector


@MODELS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 preprocess_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RetinaNet,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             preprocess_cfg, pretrained, init_cfg)
