from mmdet.models.builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class CenterNetv2(TwoStageDetector):
    """Implementation of CenterNet(Objects as Points)

    <https://arxiv.org/abs/1904.07850>.
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
        super(CenterNetv2, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)


