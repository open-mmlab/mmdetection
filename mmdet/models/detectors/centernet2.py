from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class CenterNet2(TwoStageDetector):
    """Implementation of `Probabilistic two-stage detection
        <https://arxiv.org/abs/2103.07461>`_.

    Slightly different from original models:
    the difference is:
    Stage 1: Original CenterNet2 when calculating heatmap loss,
        pos_indices can have same indices. Changed the
        pos_indices calculation. Besides, add another way of
        masking where dist2 equals zero, optional, the choice
        is turned on/off by original_dis_map.
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
