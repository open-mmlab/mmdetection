# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from .base_bbox import BaseInstanceBoxes
from .bbox_mode import register_bbox_mode

T = TypeVar('T')
DeviceType = Union[str, torch.device]


@register_bbox_mode(name='hbbox')
class HoriInstanceBoxes(BaseInstanceBoxes):
    """The horizontal box class used in MMDetection by default. The box data
    shape should be (..., 4). The last dimension indicates the coordinates of
    the left-top and right-bottom points of horizontal boxes.

    Args:
        bboxes (Tensor or np.ndarray): The box data with shape (..., 4).
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
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes)
        assert isinstance(bboxes, Tensor)
        if device is not None or dtype is not None:
            bboxes = bboxes.to(dtype=dtype, device=device)

        if isinstance(pattern, str):
            if pattern == 'cxcywh':
                bboxes = self.cxcywh_to_xyxy(bboxes)
            else:
                raise ValueError(f'invalide pattern {pattern}')

        assert bboxes.dim() >= 2
        assert bboxes.size(-1) == self._bbox_dim
        self.tensor = bboxes

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
             img_shape: Tuple[float, float],
             direction: str = 'horizontal') -> T:
        """Flip bboxes horizontally or vertically.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"

        Returns:
            T: Flipped boxes.
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
            T: Translated boxes.
        """
        bboxes = self.tensor
        assert len(distances) == 2
        bboxes = bboxes + bboxes.new_tensor(distances).repeat(2)
        return type(self)(bboxes)

    def clip(self: T, img_shape: Tuple[int, int]) -> T:
        """Clip boxes according to border.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            T: Cliped boxes.
        """
        bboxes = self.tensor
        bboxes[..., 0::2] = bboxes[:, 0::2].clamp(0, img_shape[1])
        bboxes[..., 1::2] = bboxes[:, 1::2].clamp(0, img_shape[0])
        return type(self)(bboxes)

    def rotate(self: T, center: Tuple[float, float], angle: float,
               img_shape: Tuple[int, int]):
        """Rotate all boxes.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle.
            img_shape (Tuple[int, int], Optional): image shape.
                Defaults to None.

        Returns:
            T: Rotated boxes.
        """
        bboxes = self.tensor
        rotation_matrix = bboxes.new_tensor(
            cv2.getRotationMatrix2D(angle, center, 1))

        corners = self.bbox2corner(bboxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(rotation_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        bboxes = self.corner2bbox(corners)
        if img_shape is not None:
            bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0, img_shape[1])
            bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0, img_shape[0])
        return type(self)(bboxes)

    def project(self: T,
                homography_matrix: Union[Tensor, np.ndarray],
                img_shape: Tuple[int, int] = None) -> T:
        """Geometric transformation for bbox.

        Args:
            homography_matrix (Tensor or np.ndarray):
                Shape (3, 3) for geometric transformation.
            img_shape (Tuple[int, int], optional): Image shape.
                Defaults to None.

        Returns:
            T: Converted bboxes.
        """
        bboxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = bboxes.new_tensor(homography_matrix)
        corners = self.bbox2corner(bboxes)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        bboxes = self.corner2bbox(corners)
        if img_shape is not None:
            bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0, img_shape[1])
            bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0, img_shape[0])
        return type(self)(bboxes)

    @staticmethod
    def bbox2corner(bboxes: Tensor) -> Tensor:
        """Convert bbox coordinates from (x1, y1, x2, y2) to corners ((x1, y1),
        (x2, y1), (x1, y2), (x2, y2)).

        Args:
            bboxes (Tensor): Shape (..., 4) for bboxes.
        Returns:
            Tensor: Shape (..., 4, 2) for corners.
        """
        x1, y1, x2, y2 = torch.split(bboxes, 1, dim=-1)
        corners = torch.cat([x1, y1, x2, y1, x1, y2, x2, y2], dim=-1)
        return corners.reshape(*corners.shape[:-1], 4, 2)

    @staticmethod
    def corner2bbox(corners: Tensor) -> Tensor:
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

    def rescale(self: T, rescale_factor: Tuple[float, float]) -> T:
        """Rescale boxes w.r.t. rescale_factor.

        Args:
            rescale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.

        Returns:
            T: Rescaled boxes.
        """
        bboxes = self.tensor
        assert len(rescale_factor) == 2
        bboxes = bboxes * bboxes.new_tensor(rescale_factor).repeat(2)
        return type(self)(bboxes)

    def rescale_size(self: T, rescale_factor: Tuple[float, float]) -> T:
        """Only rescale the box shape. The centers of boxes are unchanged.

        Args:
            rescale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.

        Returns:
            Tensor: Rescaled bboxes.
        """
        bboxes = self.tensor
        assert len(rescale_factor) == 2
        ctrs = (bboxes[..., 2:] + bboxes[..., :2]) / 2
        wh = bboxes[..., 2:] - bboxes[..., :2]
        rescale_factor = bboxes.new_tensor(rescale_factor)
        wh = wh * rescale_factor
        xy1 = ctrs - 0.5 * wh
        xy2 = ctrs + 0.5 * wh
        return torch.cat([xy1, xy2], dim=-1)

    def is_bboxes_inside(self, border: tuple) -> torch.BoolTensor:
        """Find bboxes as long as a part of bboxes is inside an region.

        Args:
            border (tuple): A tuple of region border. Allows input
                (x_min, y_min, x_max, y_max) or (x_max, y_max).

        Returns:
            BoolTensor: Index of the remaining bboxes.
        """
        if len(border) == 2:
            x_min, y_min, x_max, y_max = 0, 0, border[0], border[1]
        else:
            x_min, y_min, x_max, y_max = border
        bboxes = self.tensor
        return (bboxes[..., 0] < x_max) & (bboxes[..., 2] > x_min) \
            & (bboxes[..., 1] < y_max) & (bboxes[..., 3] > y_min)

    def find_inside_points(self, points: Tensor) -> torch.BoolTensor:
        """Find inside box points.

        Args:
            points (Tensor): points coordinates. has shape of (m, 2).

        Returns:
            BoolTensor: Index of inside box points. has shape of (m, n)
                where n is the length of flattened boxes.
        """
        bboxes = self.tensor
        if bboxes.dim() > 2:
            bboxes = bboxes.flatten(end_dim=-2)
        assert points.dim() == 2 and points.size(1) == 2

        bboxes = bboxes[None, :, :]
        points = points[:, None, :]
        x_min, y_min, x_max, y_max = bboxes.unbind(dim=-1)
        return (points[0] > x_min) & (points[0] < x_max) & \
            (points[1] > y_min) & (points[1] < y_max)
