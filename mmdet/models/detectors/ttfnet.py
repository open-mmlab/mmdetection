from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class TTFNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TTFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)
