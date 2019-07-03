from .two_stage import TwoStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class PanopticFPN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 semantic_segm_head=None,
                 shared_head=None,
                 pretrained=None):
        super(PanopticFPN, self).__init__(
            backbone=backbone,
            neck=neck,
            semantic_segm_head=semantic_segm_head,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
