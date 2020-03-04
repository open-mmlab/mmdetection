from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class FastRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 pretrained=None):
        super(FastRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            pretrained=pretrained)
