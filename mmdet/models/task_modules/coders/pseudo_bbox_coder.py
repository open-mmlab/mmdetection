# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, HorizontalBoxes, get_box_tensor
from .base_bbox_coder import BaseBBoxCoder


@TASK_UTILS.register_module()
class PseudoBBoxCoder(BaseBBoxCoder):
    """Pseudo bounding box coder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, bboxes: Tensor, gt_bboxes: Union[Tensor,
                                                      BaseBoxes]) -> Tensor:
        """torch.Tensor: return the given ``bboxes``"""
        gt_bboxes = get_box_tensor(gt_bboxes)
        return gt_bboxes

    def decode(self, bboxes: Tensor, pred_bboxes: Union[Tensor,
                                                        BaseBoxes]) -> Tensor:
        """torch.Tensor: return the given ``pred_bboxes``"""
        if self.use_box_type:
            pred_bboxes = HorizontalBoxes(pred_bboxes)
        return pred_bboxes
