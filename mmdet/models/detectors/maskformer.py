from ..builder import DETECTORS
from .panoptic_single_stage_segmentor import SingleStageSegmentor


@DETECTORS.register_module()
class MaskFormer(SingleStageSegmentor):
    r"""Implementation of `Per-Pixel Classification is
    NOT All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`"""

    def __init__(self,
                 backbone,
                 neck=None,
                 semantic_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskFormer, self).__init__(
            backbone=backbone,
            neck=neck,
            semantic_head=semantic_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
