# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractclassmethod, abstractproperty
from typing import List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

T = TypeVar('T')
DeviceType = Union[str, torch.device]
IndexType = Union[slice, int, list, torch.LongTensor, torch.cuda.LongTensor,
                  torch.BoolTensor, torch.cuda.BoolTensor]


class BaseBoxes(metaclass=ABCMeta):
    """The base class for 2D box modes.

    The functions of ``BaseBoxes`` lie in three fields:

    - Verify the tensor shape.
    - Support tensor-like operations.
    - Define abstract functions for 2D boxes.

    In ``__init__`` , ``BaseBoxes`` verifies the validity of the data shape
    w.r.t ``_bbox_dim``. The tensor with the dimension >= 2 and the length
    of the last dimension being ``_bbox_dim`` will be regarded as valid.
    ``BaseBoxes`` will restore them at the field ``tensor``. It's necessary
    to override ``_bbox_dim`` in subclass to guarantee the data shape is
    correct.

    There are many basic tensor-like functions implemented in ``BaseBoxes``.
    In most cases, users can operate ``BaseBoxes`` instance like a normal
    tensor. To protect the validity of data shape, All tensor-like functions
    cannot modify the last dimension of ``self.tensor``.

    When designing a new box mode, users need to inherit from ``BaseBoxes``
    and override abstract methods and specify the ``_bbox_dim``. Then,
    register the new box mode by using the decorator ``register_bbox_mode``.

    Args:
        bboxes (Tensor or np.ndarray or Sequence): The box data with shape
            (a0, a1, ..., bbox_dim).
        dtype (torch.dtype, Optional): data type of bboxes.
        device (str or torch.device, Optional): device of bboxes.
    """

    # Used to verify the last dimension length
    # Should override it in subclass.
    _bbox_dim: int = 0

    def __init__(self,
                 bboxes: Union[Tensor, np.ndarray, Sequence],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[DeviceType] = None) -> None:
        if isinstance(bboxes, (np.ndarray, Tensor, Sequence)):
            bboxes = torch.as_tensor(bboxes)
        else:
            raise TypeError('bboxes should be Tensor, ndarray, or Sequence, ',
                            f'but got {type(bboxes)}')

        if device is not None or dtype is not None:
            bboxes = bboxes.to(dtype=dtype, device=device)

        assert bboxes.dim() >= 2 and bboxes.size(-1) == self.bbox_dim, \
            ('The bboxes dimension must >= 2 and the length of the last '
             f'dimension must be {self.bbox_dim}, but get bboxes with '
             f'shape {bboxes.shape}.')
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

    def create_empty_bbox(self: T,
                          dtype: Optional[torch.dtype] = None,
                          device: Optional[DeviceType] = None) -> T:
        """Create an empty box with shape of (0, bbox_dim).

        Args:
            dtype (torch.dtype, Optional): data type of bboxes.
            device (str or torch.device, Optional): device of bboxes.

        Returns:
            T: Empty box with shape of (0, bbox_dim).
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
            sizes (Tuple[int]): The size of fake boxes. The last value must
                be equal with ``self.bbox_dim``.
            fill (float): filling value. Defaults to None.
            dtype (torch.dtype, Optional): data type of bboxes.
            device (str or torch.device, Optional): device of bboxes.

        Returns:
            T: Fake boxes with shape of ``sizes``.
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
        assert dim0 != -1 and dim0 != ndim - 1
        assert dim1 != -1 and dim1 != ndim - 1
        return type(self)(self.tensor.transpose(dim0, dim1))

    def permute(self: T, *dims: Tuple[int]) -> T:
        """Reload ``permute`` from self.tensor."""
        assert dims[-1] == -1 or dims[-1] == self.tensor.dim() - 1
        return type(self)(self.tensor.permute(dims))

    def split(self: T,
              split_size_or_sections: Union[int, Sequence[int]],
              dim: int = 0) -> List[T]:
        """Reload ``split`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        bboxes_list = self.tensor.split(split_size_or_sections, dim=dim)
        return [type(self)(bboxes) for bboxes in bboxes_list]

    def chunk(self: T, chunks: int, dim: int = 0) -> List[T]:
        """Reload ``chunk`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        bboxes_list = self.tensor.chunk(chunks, dim=dim)
        return [type(self)(bboxes) for bboxes in bboxes_list]

    def unbind(self: T, dim: int = 0) -> T:
        """Reload ``unbind`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        bboxes_list = self.tensor.unbind(dim=dim)
        return [type(self)(bboxes) for bboxes in bboxes_list]

    def flatten(self: T, start_dim: int = 0, end_dim: int = -2) -> T:
        """Reload ``flatten`` from self.tensor."""
        assert end_dim != -1 and end_dim != self.tensor.dim() - 1
        return type(self)(self.tensor.flatten(start_dim, end_dim))

    def squeeze(self: T, dim: Optional[int] = None) -> T:
        """Reload ``squeeze`` from self.tensor."""
        bboxes = self.tensor.squeeze() if dim is None else self.tensor.squeeze(
            dim)
        return type(self)(bboxes)

    def unsqueeze(self: T, dim: int) -> T:
        """Reload ``unsqueeze`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim()
        return type(self)(self.tensor.unsqueeze(dim))

    @classmethod
    def cat(cls: Type[T], bbox_list: Sequence[T], dim: int = 0) -> T:
        """Cancatenates a box instance list into one single box instance.
        Similar to ``torch.cat``.

        Args:
            bbox_list (Sequence[T]): A sequence of box instances.
            dim (int): The dimension over which the box are concatenated.
                Defaults to 0.

        Returns:
            T: Concatenated box instance.
        """
        assert isinstance(bbox_list, Sequence)
        if len(bbox_list) == 0:
            raise ValueError('bbox_list should not be a empty list.')

        assert dim != -1 and dim != bbox_list[0].dim() - 1
        assert all(isinstance(bboxes, cls) for bboxes in bbox_list)

        th_bbox_list = [bboxes.tensor for bboxes in bbox_list]
        return cls(torch.cat(th_bbox_list, dim=dim))

    @classmethod
    def stack(cls: Type[T], bbox_list: Sequence[T], dim: int = 0) -> T:
        """Concatenates a sequence of tensors along a new dimension. Similar to
        ``torch.stack``.

        Args:
            bbox_list (Sequence[T]): A sequence of box instances.
            dim (int): Dimension to insert. Defaults to 0.

        Returns:
            T: Concatenated box instance.
        """
        assert isinstance(bbox_list, Sequence)
        if len(bbox_list) == 0:
            raise ValueError('bbox_list should not be a empty list.')

        assert dim != -1 and dim != bbox_list[0].dim()
        assert all(isinstance(bboxes, cls) for bboxes in bbox_list)

        th_bbox_list = [bboxes.tensor for bboxes in bbox_list]
        return cls(torch.stack(th_bbox_list, dim=dim))

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
            T: Flipped boxes with the same shape as the original boxes.
        """
        pass

    @abstractclassmethod
    def translate(self: T, distances: Tuple[float, float]) -> T:
        """Translate bboxes.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.

        Returns:
            T: Translated boxes with the same shape as the original boxes.
        """
        pass

    @abstractclassmethod
    def clip(self: T, img_shape: Tuple[int, int]) -> T:
        """Clip boxes according to the image shape.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            T: Cliped boxes with the same shape as the original boxes.
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
            img_shape (Tuple[int, int], Optional): A tuple of image height
                and width. Defaults to None.

        Returns:
            T: Rotated boxes with the same shape as the original boxes.
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
            img_shape (Tuple[int, int], Optional): A tuple of image height
                and width. Defaults to None.

        Returns:
            T: Converted bboxes with the same shape as the original boxes.
        """
        pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
    def resize_bboxes(self: T, scale_factor: Tuple[float, float]) -> T:
        """Resize the box width and height w.r.t scale_factor.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.

        Returns:
            Tensor: Resized bboxes with the same shape as the original boxes.
        """
        pass

    @abstractclassmethod
    def is_bboxes_inside(self, img_shape: Tuple[int, int]) -> torch.BoolTensor:
        """Find bboxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.

        Returns:
            BoolTensor: Index of the remaining bboxes. Assuming the original
            boxes have shape (a0, a1, ..., bbox_dim), the output has shape
            (a0, a1, ...).
        """
        pass

    @abstractclassmethod
    def find_inside_points(self, points: Tensor) -> torch.BoolTensor:
        """Find inside box points.

        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).

        Returns:
            BoolTensor: Index of inside box points. Has shape of (m, n)
                where n is the length of the boxes after flattening.
        """
        pass
