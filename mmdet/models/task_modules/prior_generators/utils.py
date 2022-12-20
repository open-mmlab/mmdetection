# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
from torch import Tensor

from mmdet.structures.bbox import BaseBoxes


def anchor_inside_flags(flat_anchors: Tensor,
                        valid_flags: Tensor,
                        img_shape: Tuple[int],
                        allowed_border: int = 0) -> Tensor:
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        if isinstance(flat_anchors, BaseBoxes):
            inside_flags = valid_flags & \
                flat_anchors.is_inside([img_h, img_w],
                                       all_inside=True,
                                       allowed_border=allowed_border)
        else:
            inside_flags = valid_flags & \
                (flat_anchors[:, 0] >= -allowed_border) & \
                (flat_anchors[:, 1] >= -allowed_border) & \
                (flat_anchors[:, 2] < img_w + allowed_border) & \
                (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def calc_region(bbox: Tensor,
                ratio: float,
                featmap_size: Optional[Tuple] = None) -> Tuple[int]:
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4).
        ratio (float): Ratio of the output region.
        featmap_size (tuple, Optional): Feature map size in (height, width)
            order used for clipping the boundary. Defaults to None.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1])
        y1 = y1.clamp(min=0, max=featmap_size[0])
        x2 = x2.clamp(min=0, max=featmap_size[1])
        y2 = y2.clamp(min=0, max=featmap_size[0])
    return (x1, y1, x2, y2)
