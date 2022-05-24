# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmengine.config import ConfigDict

from mmdet.registry import MODELS
from .single_stage import SingleStageDetector


@MODELS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: Union[ConfigDict, dict],
                 neck: Union[ConfigDict, dict],
                 bbox_head: Union[ConfigDict, dict],
                 train_cfg: Optional[Union[ConfigDict, dict]] = None,
                 test_cfg: Optional[Union[ConfigDict, dict]] = None,
                 preprocess_cfg: Optional[Union[ConfigDict, dict]] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[Union[ConfigDict, dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            preprocess_cfg=preprocess_cfg)
