# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from .base_bbox import BaseBoxes
from .bbox_mode import register_bbox_mode

T = TypeVar('T')
DeviceType = Union[str, torch.device]


@register_bbox_mode(name='hbox')
class HorizontalBoxes(BaseBoxes):
    """The horizontal box class used in MMDetection by default.

    The ``_bbox_dim`` of ``HorizontalBoxes`` is 4, which means the input
    should have shape of (a0, a1, ..., 4). Two formats of box tensor are
    supported in ``HorizontalBoxes``:

    - 'xyxy': Each row of data indicates (x1, y1, x2, y2), which are the
      coordinates of the right-top and left-bottom points.
    - 'cxcywh': Each row of data indicates (x, y, w, h), where (x, y) are the
      coordinates of the box centers and (w, h) are the width and height.

    ``HorizontalBoxes`` only restores 'xyxy' format of data. If the format of
    the input is 'cxcywh', users need to input ``pattern='cxcywh'`` and The
    code will convert 'cxcywh' inputs to 'xyxy' automatically.

    Args:
        bboxes (Tensor or np.ndarray or Sequence): The box data with
            shape (..., 4).
        dtype (torch.dtype, Optional): data type of bboxes.
        device (str or torch.device, Optional): device of bboxes.
        pattern (str, Optional): the pattern of bboxes. If pattern is 'cxcywh',
            the `bboxes` will convert to 'xyxy' pattern. Defaults to None.
    """

    _bbox_dim: int = 4

    def __init__(self,
                 bboxes: Union[Tensor, np.ndarray],
                 dtype: torch.dtype = None,
                 device: DeviceType = None,
                 pattern: Optional[str] = None) -> None:
        super().__init__(bboxes=bboxes, dtype=dtype, device=device)
        if isinstance(pattern, str):
            if pattern not in ('xyxy', 'cxcywh'):
                raise ValueError(f'Get invalid pattern {pattern}.')
            if pattern == 'cxcywh':
                self.tensor = self.cxcywh_to_xyxy(self.tensor)

    @staticmethod
    def cxcywh_to_xyxy(bboxes: Tensor) -> Tensor:
        """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

        Args:
            bbox (Tensor): Shape (n, 4) for bboxes.

        Returns:
            Tensor: Converted bboxes.
        """
        ctr, wh = bboxes.split((2, 2), dim=-1)
        return torch.cat([(ctr - wh / 2), (ctr + wh / 2)], dim=-1)

    @staticmethod
    def xyxy_to_cxcywh(bboxes: Tensor) -> Tensor:
        """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

        Args:
            bbox (Tensor): Shape (n, 4) for bboxes.

        Returns:
            Tensor: Converted bboxes.
        """
        xy1, xy2 = bboxes.split((2, 2), dim=-1)
        return torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)

    @property
    def cxcywh(self) -> Tensor:
        """Return a tensor representing the cxcywh pattern boxes."""
        return self.xyxy_to_cxcywh(self.tensor)

    @property
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes."""
        bboxes = self.tensor
        return (bboxes[..., :2] + bboxes[..., 2:]) / 2

    @property
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes."""
        bboxes = self.tensor
        return (bboxes[..., 2] - bboxes[..., 0]) * (
            bboxes[..., 3] - bboxes[..., 1])

    @property
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes."""
        bboxes = self.tensor
        return bboxes[..., 2] - bboxes[..., 0]

    @property
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes."""
        bboxes = self.tensor
        return bboxes[..., 3] - bboxes[..., 1]

    def flip(self: T,
             img_shape: Tuple[int, int],
             direction: str = 'horizontal') -> T:
        """Flip bboxes horizontally or vertically.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"

        Returns:
            T: Flipped boxes with the same shape as the original boxes.
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        bboxes = self.tensor
        flipped = bboxes.clone()
        if direction == 'horizontal':
            flipped[..., 0] = img_shape[1] - bboxes[..., 2]
            flipped[..., 2] = img_shape[1] - bboxes[..., 0]
        elif direction == 'vertical':
            flipped[..., 1] = img_shape[0] - bboxes[..., 3]
            flipped[..., 3] = img_shape[0] - bboxes[..., 1]
        else:
            flipped[..., 0] = img_shape[1] - bboxes[..., 2]
            flipped[..., 1] = img_shape[0] - bboxes[..., 3]
            flipped[..., 2] = img_shape[1] - bboxes[..., 0]
            flipped[..., 3] = img_shape[0] - bboxes[..., 1]
        return type(self)(flipped)

    def translate(self: T, distances: Tuple[float, float]) -> T:
        """Translate bboxes.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.

        Returns:
            T: Translated boxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        assert len(distances) == 2
        bboxes = bboxes + bboxes.new_tensor(distances).repeat(2)
        return type(self)(bboxes)

    def clip(self: T, img_shape: Tuple[int, int]) -> T:
        """Clip boxes according to the image shape.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            T: Cliped boxes with the same shape as the original boxes.
        """
        bboxes = self.tensor.clone()
        bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0, img_shape[1])
        bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0, img_shape[0])
        return type(self)(bboxes)

    def rotate(self: T,
               center: Tuple[float, float],
               angle: float,
               img_shape: Optional[Tuple[int, int]] = None) -> T:
        """Rotate all boxes.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle.
            img_shape (Tuple[int, int], Optional): A tuple of image height
                and width. Defaults to None.

        Returns:
            T: Rotated boxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        rotation_matrix = bboxes.new_tensor(
            cv2.getRotationMatrix2D(center, angle, 1))

        corners = self.hbbox2corner(bboxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(rotation_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        bboxes = self.corner2hbbox(corners)
        if img_shape is not None:
            bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0, img_shape[1])
            bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0, img_shape[0])
        return type(self)(bboxes)

    def project(self: T,
                homography_matrix: Union[Tensor, np.ndarray],
                img_shape: Optional[Tuple[int, int]] = None) -> T:
        """Geometric transformation for bbox.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
            img_shape (Tuple[int, int], Optional): A tuple of image height
                and width. Defaults to None.

        Returns:
            T: Converted bboxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = bboxes.new_tensor(homography_matrix)
        corners = self.hbbox2corner(bboxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        bboxes = self.corner2hbbox(corners)
        if img_shape is not None:
            bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0, img_shape[1])
            bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0, img_shape[0])
        return type(self)(bboxes)

    @staticmethod
    def hbbox2corner(bboxes: Tensor) -> Tensor:
        """Convert bbox coordinates from (x1, y1, x2, y2) to corners ((x1, y1),
        (x2, y1), (x1, y2), (x2, y2)).

        Args:
            bboxes (Tensor): Shape (..., 4) for bboxes.

        Returns:
            Tensor: Shape (..., 4, 2) for corners.
        """
        x1, y1, x2, y2 = torch.split(bboxes, 1, dim=-1)
        corners = torch.cat([x1, y1, x2, y1, x1, y2, x2, y2], dim=-1)
        return corners.view(*corners.shape[:-1], 4, 2)

    @staticmethod
    def corner2hbbox(corners: Tensor) -> Tensor:
        """Convert bbox coordinates from corners ((x1, y1), (x2, y1), (x1, y2),
        (x2, y2)) to (x1, y1, x2, y2).

        Args:
            corners (Tensor): Shape (..., 4, 2) for corners.

        Returns:
            Tensor: Shape (..., 4) for bboxes.
        """
        min_xy = corners.min(dim=-2)[0]
        max_xy = corners.max(dim=-2)[0]
        return torch.cat([min_xy, max_xy], dim=-1)

    def rescale(self: T,
                scale_factor: Tuple[float, float],
                mapping_back=False) -> T:
        """Rescale boxes w.r.t. rescale_factor.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
            mapping_back (bool): Mapping back the rescaled bboxes.
                Defaults to False.

        Returns:
            T: Rescaled boxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        assert len(scale_factor) == 2
        scale_factor = bboxes.new_tensor(scale_factor).repeat(2)
        bboxes = bboxes / scale_factor if mapping_back else \
            bboxes * scale_factor
        return type(self)(bboxes)

    def resize_bboxes(self: T, scale_factor: Tuple[float, float]) -> T:
        """Resize the box width and height w.r.t scale_factor.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.

        Returns:
            Tensor: Resized bboxes with the same shape as the original boxes.
        """
        bboxes = self.tensor
        assert len(scale_factor) == 2
        ctrs = (bboxes[..., 2:] + bboxes[..., :2]) / 2
        wh = bboxes[..., 2:] - bboxes[..., :2]
        scale_factor = bboxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        xy1 = ctrs - 0.5 * wh
        xy2 = ctrs + 0.5 * wh
        bboxes = torch.cat([xy1, xy2], dim=-1)
        return type(self)(bboxes)

    def is_bboxes_inside(self, img_shape: Tuple[int, int]) -> torch.BoolTensor:
        """Find bboxes inside the image.

        In ``HorizontalBoxes``, as long as a part of the box is inside the
        image, this box will be regarded as True.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            BoolTensor: Index of the remaining bboxes. Assuming the original
            boxes have shape (a0, a1, ..., bbox_dim), the output has shape
            (a0, a1, ...).
        """
        img_h, img_w = img_shape
        bboxes = self.tensor
        return (bboxes[..., 0] < img_w) & (bboxes[..., 2] > 0) \
            & (bboxes[..., 1] < img_h) & (bboxes[..., 3] > 0)

    def find_inside_points(self, points: Tensor) -> torch.BoolTensor:
        """Find inside box points.

        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).

        Returns:
            BoolTensor: Index of inside box points. Has shape of (m, n)
                where n is the length of the boxes after flattening.
        """
        bboxes = self.tensor
        if bboxes.dim() > 2:
            bboxes = bboxes.flatten(end_dim=-2)

        bboxes = bboxes[None, :, :]
        points = points[:, None, :]
        x_min, y_min, x_max, y_max = bboxes.unbind(dim=-1)
        return (points[..., 0] > x_min) & (points[..., 0] < x_max) & \
            (points[..., 1] > y_min) & (points[..., 1] < y_max)
