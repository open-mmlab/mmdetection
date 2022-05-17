# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .single_stage import SingleStageDetector


@MODELS.register_module()
class NASFCOS(SingleStageDetector):
    """NAS-FCOS: Fast Neural Architecture Search for Object Detection.

    https://arxiv.org/abs/1906.0442
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 preprocess_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(NASFCOS, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            preprocess_cfg=preprocess_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
