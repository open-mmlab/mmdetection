from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class CenterNet2(TwoStageDetector):
    """Implementation of ` <>`_

    Slightly different from original models:
    the differences includes:
    ...

    For openmmdet, Currently the detector, including CenterNet2Head
    DOES NOT support onnix export.

    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterNet2, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.init_weights()


