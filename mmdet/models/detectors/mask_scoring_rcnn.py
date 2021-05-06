from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class MaskScoringRCNN(TwoStageDetector):
    """Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskScoringRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
