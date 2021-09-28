from ..builder import DETECTORS
from .single_stage_instance_seg import SingleStageInstanceSegmentor


@DETECTORS.register_module()
class SOLO(SingleStageInstanceSegmentor):
    """`SOLO: Segmenting Objects by Locations
    <https://arxiv.org/abs/1912.04488>`_

    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
