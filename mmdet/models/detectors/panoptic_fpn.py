# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .panoptic_two_stage_segmentor import TwoStagePanopticSegmentor


@DETECTORS.register_module()
class PanopticFPN(TwoStagePanopticSegmentor):
    r"""Implementation of `Panoptic feature pyramid
    networks <https://arxiv.org/pdf/1901.02446>`_"""

    def __init__(
            self,
            backbone,
            neck=None,
            rpn_head=None,
            roi_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None,
            # for panoptic segmentation
            semantic_head=None,
            panoptic_fusion_head=None):
        super(PanopticFPN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            semantic_head=semantic_head,
            panoptic_fusion_head=panoptic_fusion_head)
