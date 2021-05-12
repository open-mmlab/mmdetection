from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class PAA(SingleStageDetector):
    """Implementation of `PAA <https://arxiv.org/pdf/2007.08103.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 aug_bbox_post_processes=[
                     dict(type='MergeResults'),
                     dict(
                         type='NaiveNMS',
                         iou_threshold=0.5,
                         class_agnostic=False,
                         max_num=100)
                 ]):
        super(PAA, self).__init__(
            backbone,
            neck,
            bbox_head,
            train_cfg,
            test_cfg,
            pretrained,
            init_cfg,
            aug_bbox_post_processes=aug_bbox_post_processes)
