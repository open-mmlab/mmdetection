from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class PointRend(TwoStageDetector):
    """PointRend: Image Segmentation as Rendering

    This detector is the implementation of
    `PointRend <https://arxiv.org/abs/1912.08193>`_.

    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(PointRend, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
