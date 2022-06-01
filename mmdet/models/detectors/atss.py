# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import ConfigType, OptConfigType, OptMultiConfig
from mmdet.registry import MODELS
from .single_stage import SingleStageDetector


@MODELS.register_module()
class ATSS(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 preprocess_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            preprocess_cfg=preprocess_cfg,
            init_cfg=init_cfg)
