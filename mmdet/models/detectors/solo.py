from ..builder import DETECTORS
from .single_stage_seg import SingleStageSegDetector


@DETECTORS.register_module()
class SOLO(SingleStageSegDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SOLO, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
