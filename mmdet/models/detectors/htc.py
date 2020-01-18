import torch
import torch.nn.functional as F

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from .. import builder
from ..registry import DETECTORS
from .cascade_rcnn import CascadeRCNN


@DETECTORS.register_module
class HybridTaskCascade(CascadeRCNN):

    def __init__(self, **kwargs):
        super(HybridTaskCascade, self).__init__(**kwargs)

    @property
    def with_semantic(self):
        return self.roi_head.with_semantic
