# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.utils.misc import get_box_tensor
from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import HorizontalBoxes
from .base_bbox_coder import BaseBBoxCoder


@TASK_UTILS.register_module()
class PseudoBBoxCoder(BaseBBoxCoder):
    """Pseudo bounding box coder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, bboxes, gt_bboxes):
        """torch.Tensor: return the given ``bboxes``"""
        gt_bboxes = get_box_tensor(gt_bboxes)
        return gt_bboxes

    def decode(self, bboxes, pred_bboxes):
        """torch.Tensor: return the given ``pred_bboxes``"""
        if self.use_box_type:
            pred_bboxes = HorizontalBoxes(pred_bboxes)
        return pred_bboxes
