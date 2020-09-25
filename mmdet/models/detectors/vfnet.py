from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class VFNet(SingleStageDetector):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)
