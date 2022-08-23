# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import TASK_UTILS
from .base_bbox_coder import BaseBBoxCoder
from mmdet.structures.bbox import HorizontalBoxes


@TASK_UTILS.register_module()
class PseudoBBoxCoder(BaseBBoxCoder):
    """Pseudo bounding box coder."""

    def __init__(self, **kwargs):
        super(BaseBBoxCoder, self).__init__(**kwargs)

    def encode(self, bboxes, gt_bboxes):
        """torch.Tensor: return the given ``bboxes``"""
        return gt_bboxes

    def decode(self, bboxes, pred_bboxes):
        """torch.Tensor: return the given ``pred_bboxes``"""
        if self.with_boxlist:
            pred_bboxes = HorizontalBoxes(pred_bboxes)
        return pred_bboxes
