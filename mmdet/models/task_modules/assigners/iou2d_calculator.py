# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, bbox_overlaps, get_box_tensor


def cast_tensor_type(x: Tensor,
                     scale: float = 1.,
                     dtype: Optional[torch.dtype] = None) -> Tensor:
    """Convert Tensor to float16."""
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@TASK_UTILS.register_module()
class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self,
                 scale: float = 1.,
                 dtype: Optional[torch.dtype] = None) -> None:
        self.scale = scale
        self.dtype = dtype

    def __call__(self,
                 bboxes1: Union[Tensor, BaseBoxes, InstanceData],
                 bboxes2: Union[Tensor, BaseBoxes, InstanceData],
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 key1: Optional[str] = None,
                 key2: Optional[str] = None) -> Tensor:
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor or :obj:`BaseBoxes` or :obj:`InstanceData`): bboxes
                have shape (m, 4) in <x1, y1, x2, y2> format, or shape (m, 5)
                in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor or :obj:`BaseBoxes` or :obj:`InstanceData`): bboxes
                have shape (m, 4) in <x1, y1, x2, y2> format, or shape (m, 5)
                in <x1, y1, x2, y2, score> format, or be empty.
                If ``is_aligned `` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union). Defaults to 'iou'.
            is_aligned (bool, optional): If True, then m and n must be equal.
                Defaults to False.
            key1 (str, optional): The key to get bboxes1, is necessary when
                bboxes1 is :obj:`InstanceData`. Defaults to None.
            key2 (str, optional): The key to get bboxes2, is necessary when
                bboxes2 is :obj:`InstanceData`. Defaults to None.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        if isinstance(bboxes1, InstanceData):
            assert key1 is not None
            bboxes1 = bboxes1.get(key1)
        if isinstance(bboxes2, InstanceData):
            assert key2 is not None
            bboxes2 = bboxes2.get(key2)

        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str
