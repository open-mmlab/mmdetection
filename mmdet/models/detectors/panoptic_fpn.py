# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .panoptic_two_stage_segmentor import TwoStagePanopticSegmentor


@MODELS.register_module()
class PanopticFPN(TwoStagePanopticSegmentor):
    r"""Implementation of `Panoptic feature pyramid
    networks <https://arxiv.org/pdf/1901.02446>`_"""

    def __init__(
            self,
            backbone: ConfigType,
            neck: OptConfigType = None,
            rpn_head: OptConfigType = None,
            roi_head: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            # for panoptic segmentation
            semantic_head: OptConfigType = None,
            panoptic_fusion_head: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            semantic_head=semantic_head,
            panoptic_fusion_head=panoptic_fusion_head)
