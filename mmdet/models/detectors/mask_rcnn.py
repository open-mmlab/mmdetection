# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmengine.config import ConfigDict

from mmdet.registry import MODELS
from .two_stage import TwoStageDetector


@MODELS.register_module()
class MaskRCNN(TwoStageDetector):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone: Union[ConfigDict, dict],
                 rpn_head: Union[ConfigDict, dict],
                 roi_head: Union[ConfigDict, dict],
                 train_cfg: Union[ConfigDict, dict],
                 test_cfg: Union[ConfigDict, dict],
                 neck: Optional[Union[ConfigDict, dict]] = None,
                 pretrained: Optional[str] = None,
                 preprocess_cfg: Optional[Union[ConfigDict, dict]] = None,
                 init_cfg: Optional[Union[ConfigDict, dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            preprocess_cfg=preprocess_cfg)
