from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class FCOS(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
