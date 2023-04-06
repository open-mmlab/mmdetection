# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import (BaseBoxes, HorizontalBoxes, bbox2distance,
                                   distance2bbox, get_box_tensor)
from .base_bbox_coder import BaseBBoxCoder


@TASK_UTILS.register_module()
class DistancePointBBoxCoder(BaseBBoxCoder):
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border: Optional[bool] = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_border = clip_border

    def encode(self,
               points: Tensor,
               gt_bboxes: Union[Tensor, BaseBoxes],
               max_dis: Optional[float] = None,
               eps: float = 0.1) -> Tensor:
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor or :obj:`BaseBoxes`): Shape (N, 4), The format
                is "xyxy"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4).
        """
        gt_bboxes = get_box_tensor(gt_bboxes)
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return bbox2distance(points, gt_bboxes, max_dis, eps)

    def decode(
        self,
        points: Tensor,
        pred_bboxes: Tensor,
        max_shape: Optional[Union[Sequence[int], Tensor,
                                  Sequence[Sequence[int]]]] = None
    ) -> Union[Tensor, BaseBoxes]:
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom). Shape (B, N, 4)
                or (N, 4)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Union[Tensor, :obj:`BaseBoxes`]: Boxes with shape (N, 4) or
            (B, N, 4)
        """
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_border is False:
            max_shape = None
        bboxes = distance2bbox(points, pred_bboxes, max_shape)

        if self.use_box_type:
            bboxes = HorizontalBoxes(bboxes)
        return bboxes
