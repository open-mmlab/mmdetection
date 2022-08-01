# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractclassmethod, abstractproperty
from typing import List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

T = TypeVar('T')
DeviceType = Union[str, torch.device]
IndexType = Union[slice, int, list, torch.LongTensor, torch.cuda.LongTensor,
                  torch.BoolTensor, torch.cuda.BoolTensor]


class BaseInstanceBoxes(metaclass=ABCMeta):
    """BaseInstanceBoxes is a base class that defines the data shape and some
    commonly used abstract methods of 2D boxes. Basic tensor-like functions are
    implemented in BaseInstanceBoxes so that users can treat its instance as a
    normal tensor in most cases.

    When initializing a box instance, BaseInstanceBoxes will verify the
    validity of box data shape w.r.t the class attribute ``_bbox_dim``.
    The tensor with the dimension >= 2 and the length of the last dimension
    being ``_bbox_dim`` is regarded as the valid box tensor. BaseInstance
    restores the data tensor at the field ``tensor``.

    Args:
        bboxes (Tensor or np.ndarray or Sequence): The box data with shape
            (..., bbox_dim).
        dtype (torch.dtype, Optional): data type of bboxes.
        device (str or torch.device, Optional): device of bboxes.
    """

    _bbox_dim: int = 0

    def __init__(self,
                 bboxes: Union[Tensor, np.ndarray, Sequence],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[DeviceType] = None) -> None:
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes)
        elif not isinstance(bboxes, Tensor):
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        assert isinstance(bboxes, Tensor)
        if device is not None or dtype is not None:
            bboxes = bboxes.to(dtype=dtype, device=device)

        assert bboxes.dim() >= 2
        assert bboxes.size(-1) == self.bbox_dim
        self.tensor = bboxes

    def convert_to(self, dst_mode: Union[str, type]):
        """Convert self to another box mode.

        Args:
            dst_mode (str or type): destination box mode.

        Returns:
            object: destination box mode object .
        """
        from .bbox_mode import convert_bbox_mode
        return convert_bbox_mode(self, dst_mode=dst_mode)

    @property
    def bbox_dim(self) -> int:
        """Return the value of ``self._bbox_dim``"""
        return self._bbox_dim

    def create_empty_bboxes(self: T,
                            dtype: Optional[torch.dtype] = None,
                            device: Optional[DeviceType] = None) -> T:
        """Create a empty box with shape of (0, bbox_dim).

        Args:
            dtype (torch.dtype, Optional): data type of bboxes.
            device (str or torch.device, Optional): device of bboxes.

        Returns:
            T: Empty box instance.
        """
        empty_bboxes = self.tensor.new_zeros((0, self._bbox_dim),
                                             dtype=dtype,
                                             device=device)
        return type(self)(empty_bboxes)

    def create_fake_bboxes(self: T,
                           sizes: Tuple[int],
                           fill: float = 0,
                           dtype: Optional[torch.dtype] = None,
                           device: Optional[DeviceType] = None) -> T:
        """Create specific size fake boxes.

        Args:
            sizes (Tuple[int]): The size of fake boxes.
            fill (float): filling value.
            dtype (torch.dtype, Optional): data type of bboxes.
            device (str or torch.device, Optional): device of bboxes.

        Returns:
            T: Empty box instance.
        """
        fake_bboxes = self.tensor.new_full(
            sizes, fill, dtype=dtype, device=device)
        return type(self)(fake_bboxes)

    def __getitem__(self: T, index: IndexType) -> T:
        """Rewrite getitem to protect the last dimension shape."""
        bboxes = self.tensor
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < bboxes.dim()
        elif isinstance(index, tuple):
            assert len(index) < bboxes.dim()
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        bboxes = bboxes[index]
        if bboxes.dim() == 1:
            bboxes = bboxes.reshape(1, -1)
        return type(self)(bboxes)

    def __setitem__(self: T, index: IndexType, values: Union[Tensor, T]) -> T:
        """Rewrite setitem to protect the last dimension shape."""
        assert type(values) is type(self)
        values = values.tensor

        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < self.tensor.dim()
        elif isinstance(index, tuple):
            assert len(index) < self.tensor.dim()
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        self.tensor[index] = values

    def __len__(self) -> int:
        """Return the length of self.tensor first dimension."""
        return self.tensor.size(0)

    def __repr__(self) -> str:
        """Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n' + str(self.tensor) + ')'

    def new_tensor(self, *args, **kwargs) -> Tensor:
        """Reload ``new_tensor`` from self.tensor."""
        return self.tensor.new_tensor(*args, **kwargs)

    def new_full(self, *args, **kwargs) -> Tensor:
        """Reload ``new_full`` from self.tensor."""
        return self.tensor.new_full(*args, **kwargs)

    def new_empty(self, *args, **kwargs) -> Tensor:
        """Reload ``new_empty`` from self.tensor."""
        return self.tensor.new_empty(*args, **kwargs)

    def new_ones(self, *args, **kwargs) -> Tensor:
        """Reload ``new_ones`` from self.tensor."""
        return self.tensor.new_ones(*args, **kwargs)

    def new_zeros(self, *args, **kwargs) -> Tensor:
        """Reload ``new_zeros`` from self.tensor."""
        return self.tensor.new_zeros(*args, **kwargs)

    def size(self, dim: Optional[int] = None) -> Union[int, torch.Size]:
        """Reload new_zeros from self.tensor."""
        # self.tensor.size(dim) cannot work when dim=None.
        return self.tensor.size() if dim is None else self.tensor.size(dim)

    def dim(self) -> int:
        """Reload ``dim`` from self.tensor."""
        return self.tensor.dim()

    @property
    def device(self) -> torch.device:
        """Reload ``device`` from self.tensor."""
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """Reload ``dtype`` from self.tensor."""
        return self.tensor.dtype

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self: T, *args, **kwargs) -> T:
        """Reload ``to`` from self.tensor."""
        return type(self)(self.tensor.to(*args, **kwargs))

    def cpu(self: T) -> T:
        """Reload ``cpu`` from self.tensor."""
        return type(self)(self.tensor.cpu())

    def cuda(self: T, *args, **kwargs) -> T:
        """Reload ``cuda`` from self.tensor."""
        return type(self)(self.tensor.cuda(*args, **kwargs))

    def clone(self: T) -> T:
        """Reload ``clone`` from self.tensor."""
        return type(self)(self.tensor.clone())

    def detach(self: T) -> T:
        """Reload ``detach`` from self.tensor."""
        return type(self)(self.tensor.detach())

    def view(self: T, *shape: Tuple[int]) -> T:
        """Reload ``view`` from self.tensor."""
        return type(self)(self.tensor.view(shape))

    def expand(self: T, *sizes: Tuple[int]) -> T:
        """Reload ``expand`` from self.tensor."""
        return type(self)(self.tensor.expand(sizes))

    def repeat(self: T, *sizes: Tuple[int]) -> T:
        """Reload ``repeat`` from self.tensor."""
        return type(self)(self.tensor.repeat(sizes))

    def transpose(self: T, dim0: int, dim1: int) -> T:
        """Reload ``transpose`` from self.tensor."""
        ndim = self.tensor.dim()
        assert dim0 not in (-1, ndim - 1)
        assert dim1 not in (-1, ndim - 1)
        return type(self)(self.tensor.transpose(dim0, dim1))

    def permute(self: T, *dims: Tuple[int]) -> T:
        """Reload ``permute`` from self.tensor."""
        assert dims[-1] in (-1, self.tensor.dim() - 1)
        return type(self)(self.tensor.permute(dims))

    def split(self: T,
              split_size_or_sections: Union[int, Sequence[int]],
              dim: int = 0) -> List[T]:
        """Reload ``split`` from self.tensor."""
        assert dim not in (-1, self.tensor.dim() - 1)
        bboxes_list = self.tensor.split(split_size_or_sections, dim=dim)
        return [type(self)(bboxes) for bboxes in bboxes_list]

    def chunk(self: T, chunks: int, dim: int = 0) -> List[T]:
        """Reload ``chunk`` from self.tensor."""
        assert dim not in (-1, self.tensor.dim() - 1)
        bboxes_list = self.tensor.chunk(chunks, dim=dim)
        return [type(self)(bboxes) for bboxes in bboxes_list]

    def unbind(self: T, dim: int = 0) -> T:
        """Reload ``unbind`` from self.tensor."""
        assert dim not in (-1, self.tensor.dim() - 1)
        bboxes_list = self.tensor.unbind(dim=dim)
        return [type(self)(bboxes) for bboxes in bboxes_list]

    def flatten(self: T, start_dim: int = 0, end_dim: int = -2) -> T:
        """Reload ``flatten`` from self.tensor."""
        assert end_dim not in (-1, self.tensor.dim() - 1)
        return type(self)(self.tensor.flatten(start_dim, end_dim))

    def squeeze(self: T, dim: Optional[int] = None) -> T:
        """Reload ``squeeze`` from self.tensor."""
        bboxes = self.tensor.squeeze() if dim is None else self.tensor.squeeze(
            dim)
        return type(self)(bboxes)

    def unsqueeze(self: T, dim: int) -> T:
        """Reload ``unsqueeze`` from self.tensor."""
        assert dim not in (-1, self.tensor.dim())
        return type(self)(self.tensor.unsqueeze(dim))

    @abstractproperty
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes."""
        pass

    @abstractproperty
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes."""
        pass

    @abstractproperty
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes."""
        pass

    @abstractproperty
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes."""
        pass

    @abstractclassmethod
    def flip(self: T,
             img_shape: Tuple[int, int],
             direction: str = 'horizontal') -> T:
        """Flip bboxes horizontally or vertically.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"

        Returns:
            T: Flipped boxes.
        """
        pass

    @abstractclassmethod
    def translate(self: T, distances: Tuple[float, float]) -> T:
        """Translate bboxes.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.

        Returns:
            T: Translated boxes.
        """
        pass

    @abstractclassmethod
    def clip(self: T, img_shape: Tuple[int, int]) -> T:
        """Clip boxes according to border.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            T: Cliped boxes.
        """
        pass

    @abstractclassmethod
    def rotate(self: T,
               center: Tuple[float, float],
               angle: float,
               img_shape: Optional[Tuple[int, int]] = None) -> T:
        """Rotate all boxes.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle.
            img_shape (Tuple[int, int], Optional): image shape.
                Defaults to None.

        Returns:
            T: Rotated boxes.
        """
        pass

    @abstractclassmethod
    def project(self: T,
                homography_matrix: Union[Tensor, np.ndarray],
                img_shape: Optional[Tuple[int, int]] = None) -> T:
        """Geometric transformation for bbox.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
            img_shape (Tuple[int, int], optional): Image shape.
                Defaults to None.

        Returns:
            T: Converted bboxes.
        """
        pass

    @abstractclassmethod
    def rescale(self: T, rescale_factor: Tuple[float, float]) -> T:
        """Rescale boxes w.r.t. rescale_factor.

        Args:
            rescale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.

        Returns:
            T: Rescaled boxes.
        """
        pass

    @abstractclassmethod
    def rescale_size(self: T, rescale_factor: Tuple[float, float]) -> T:
        """Only rescale the box shape. The centers of boxes are unchanged.

        Args:
            rescale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.

        Returns:
            Tensor: Rescaled bboxes.
        """
        pass

    @abstractclassmethod
    def is_bboxes_inside(self, border: tuple) -> torch.BoolTensor:
        """Find bboxes as long as a part of bboxes is inside an region.

        Args:
            border (tuple): A tuple of region border. Allows input
                (x_min, y_min, x_max, y_max) or (x_max, y_max).

        Returns:
            BoolTensor: Index of the remaining bboxes.
        """
        pass

    @abstractclassmethod
    def find_inside_points(self, points: Tensor) -> torch.BoolTensor:
        """Find inside box points.

        Args:
            points (Tensor): points coordinates. has shape of (m, 2).

        Returns:
            BoolTensor: Index of inside box points. has shape of (m, n)
                where n is the length of flattened boxes.
        """
        pass
