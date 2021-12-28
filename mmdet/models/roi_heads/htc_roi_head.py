# Copyright (c) OpenMMLab. All rights reserved.

from ..builder import HEADS
from .r3_roi_head import R3RoIHead


@HEADS.register_module()
class HybridTaskCascadeRoIHead(R3RoIHead):
    """Hybrid task cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1901.07518
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 **kwargs):
        super(HybridTaskCascadeRoIHead, self).__init__(
            stages=[i for i in range(num_stages)],
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            num_stages_test=None,
            mask_iou_head=None,
            ret_intermediate_results=False,
            **kwargs)
