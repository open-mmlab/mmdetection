from ..builder import DETECTORS
from .panoptic_two_stage_segmentor import PanopticTwoStageSegmentor


@DETECTORS.register_module()
class PanopticFPN(PanopticTwoStageSegmentor):
    """Implementation of Panoptic FPN."""

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
            stuff_head=None,
            panoptic_fusion_head=None,
            num_things_classes=80,
            num_stuff_classes=53):
        super(PanopticFPN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            stuff_head=stuff_head,
            panoptic_fusion_head=panoptic_fusion_head,
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes)
