# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BBOX_CODERS
from ..transforms import bbox2distance, distance2bbox
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DistancePointBBoxCoder(BaseBBoxCoder):
    """Distance Point BBox coder.

    该类可以将[x1, y1, x2, y2]格式 转为[top, bottom, left, right]格式
        同时也可以将其转回原来格式.

    Args:
        clip_border (bool, optional): 是否限制图像边界外的box. Defaults to True.
    """

    def __init__(self, clip_border=True):
        super(BaseBBoxCoder, self).__init__()
        self.clip_border = clip_border

    def encode(self, points, gt_bboxes, max_dis=None, eps=0.1):
        """将[x1, y1, x2, y2] -> [top, bottom, left, right].

        Args:
            points (Tensor): Shape [N, 2], The format is [x, y].
            gt_bboxes (Tensor): Shape [N, 4], 格式为[x1, y1, x2, y2]
            max_dis (float): 距离上限. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return bbox2distance(points, gt_bboxes, max_dis, eps)

    def decode(self, points, pred_bboxes, max_shape=None):
        """将预测的四个方向距离[left, top, right, bottom]转换成[x1, y1, x2, y2].

        Args:
            points (Tensor): Shape [B, N, 2] or [N, 2].
            pred_bboxes (Tensor): 从points到4个边界的距离[left, top, right, bottom].
                Shape [B, N, 4] or [N, 4]
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): box的边界, 指定格式为 [H, W, C] 或 [H, W].
                如果 points 形状是 (B, N, 4), 那么 max_shape 应该是 [[int,],] * B
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 4) or (B, N, 4)
        """
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_border is False:
            max_shape = None
        return distance2bbox(points, pred_bboxes, max_shape)
