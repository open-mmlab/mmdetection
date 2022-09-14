# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod, abstractproperty, abstractstaticmethod
from typing import List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch import BoolTensor, Tensor

from mmdet.structures.mask.structures import BitmapMasks, PolygonMasks

T = TypeVar('T')
DeviceType = Union[str, torch.device]
IndexType = Union[slice, int, list, torch.LongTensor, torch.cuda.LongTensor,
                  torch.BoolTensor, torch.cuda.BoolTensor, np.ndarray]
MaskType = Union[BitmapMasks, PolygonMasks]


class BaseBoxes(metaclass=ABCMeta):
    """The base class for 2D box types.

    The functions of ``BaseBoxes`` lie in three fields:

    - Verify the boxes shape.
    - Support tensor-like operations.
    - Define abstract functions for 2D boxes.

    In ``__init__`` , ``BaseBoxes`` verifies the validity of the data shape
    w.r.t ``box_dim``. The tensor with the dimension >= 2 and the length
    of the last dimension being ``box_dim`` will be regarded as valid.
    ``BaseBoxes`` will restore them at the field ``tensor``. It's necessary
    to override ``box_dim`` in subclass to guarantee the data shape is
    correct.

    There are many basic tensor-like functions implemented in ``BaseBoxes``.
    In most cases, users can operate ``BaseBoxes`` instance like a normal
    tensor. To protect the validity of data shape, All tensor-like functions
    cannot modify the last dimension of ``self.tensor``.

    When creating a new box type, users need to inherit from ``BaseBoxes``
    and override abstract methods and specify the ``box_dim``. Then, register
    the new box type by using the decorator ``register_box_type``.

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape
            (..., box_dim).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    # Used to verify the last dimension length
    # Should override it in subclass.
    box_dim: int = 0

    def __init__(self,
                 data: Union[Tensor, np.ndarray, Sequence],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[DeviceType] = None,
                 clone: bool = True) -> None:
        if isinstance(data, (np.ndarray, Tensor, Sequence)):
            data = torch.as_tensor(data)
        else:
            raise TypeError('boxes should be Tensor, ndarray, or Sequence, ',
                            f'but got {type(data)}')

        if device is not None or dtype is not None:
            data = data.to(dtype=dtype, device=device)
        # Clone the data to avoid potential bugs
        if clone:
            data = data.clone()
        # handle the empty input like []
        if data.numel() == 0:
            data = data.reshape((-1, self.box_dim))

        assert data.dim() >= 2 and data.size(-1) == self.box_dim, \
            ('The boxes dimension must >= 2 and the length of the last '
             f'dimension must be {self.box_dim}, but got boxes with '
             f'shape {data.shape}.')
        self.tensor = data

    def convert_to(self, dst_type: Union[str, type]) -> 'BaseBoxes':
        """Convert self to another box type.

        Args:
            dst_type (str or type): destination box type.

        Returns:
            :obj:`BaseBoxes`: destination box type object .
        """
        from .box_type import convert_box_type
        return convert_box_type(self, dst_type=dst_type)

    def empty_boxes(self: T,
                    dtype: Optional[torch.dtype] = None,
                    device: Optional[DeviceType] = None) -> T:
        """Create empty box.

        Args:
            dtype (torch.dtype, Optional): data type of boxes.
            device (str or torch.device, Optional): device of boxes.

        Returns:
            T: empty boxes with shape of (0, box_dim).
        """
        empty_box = self.tensor.new_zeros(
            0, self.box_dim, dtype=dtype, device=device)
        return type(self)(empty_box, clone=False)

    def fake_boxes(self: T,
                   sizes: Tuple[int],
                   fill: float = 0,
                   dtype: Optional[torch.dtype] = None,
                   device: Optional[DeviceType] = None) -> T:
        """Create fake boxes with specific sizes and fill values.

        Args:
            sizes (Tuple[int]): The size of fake boxes. The last value must
                be equal with ``self.box_dim``.
            fill (float): filling value. Defaults to 0.
            dtype (torch.dtype, Optional): data type of boxes.
            device (str or torch.device, Optional): device of boxes.

        Returns:
            T: Fake boxes with shape of ``sizes``.
        """
        fake_boxes = self.tensor.new_full(
            sizes, fill, dtype=dtype, device=device)
        return type(self)(fake_boxes, clone=False)

    def __getitem__(self: T, index: IndexType) -> T:
        """Rewrite getitem to protect the last dimension shape."""
        boxes = self.tensor
        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.device)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < boxes.dim()
        elif isinstance(index, tuple):
            assert len(index) < boxes.dim()
            # `Ellipsis`(...) is commonly used in index like [None, ...].
            # When `Ellipsis` is in index, it must be the last item.
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        boxes = boxes[index]
        if boxes.dim() == 1:
            boxes = boxes.reshape(1, -1)
        return type(self)(boxes, clone=False)

    def __setitem__(self: T, index: IndexType, values: Union[Tensor, T]) -> T:
        """Rewrite setitem to protect the last dimension shape."""
        assert type(values) is type(self), \
            'The value to be set must be the same box type as self'
        values = values.tensor

        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.device)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < self.tensor.dim()
        elif isinstance(index, tuple):
            assert len(index) < self.tensor.dim()
            # `Ellipsis`(...) is commonly used in index like [None, ...].
            # When `Ellipsis` is in index, it must be the last item.
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        self.tensor[index] = values

    def __len__(self) -> int:
        """Return the length of self.tensor first dimension."""
        return self.tensor.size(0)

    def __deepcopy__(self, memo):
        """Only clone the ``self.tensor`` when applying deepcopy."""
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other
        other.tensor = self.tensor.clone()
        return other

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

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def numel(self) -> int:
        """Reload ``numel`` from self.tensor."""
        return self.tensor.numel()

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self: T, *args, **kwargs) -> T:
        """Reload ``to`` from self.tensor."""
        return type(self)(self.tensor.to(*args, **kwargs), clone=False)

    def cpu(self: T) -> T:
        """Reload ``cpu`` from self.tensor."""
        return type(self)(self.tensor.cpu(), clone=False)

    def cuda(self: T, *args, **kwargs) -> T:
        """Reload ``cuda`` from self.tensor."""
        return type(self)(self.tensor.cuda(*args, **kwargs), clone=False)

    def clone(self: T) -> T:
        """Reload ``clone`` from self.tensor."""
        return type(self)(self.tensor)

    def detach(self: T) -> T:
        """Reload ``detach`` from self.tensor."""
        return type(self)(self.tensor.detach(), clone=False)

    def view(self: T, *shape: Tuple[int]) -> T:
        """Reload ``view`` from self.tensor."""
        return type(self)(self.tensor.view(shape), clone=False)

    def reshape(self: T, *shape: Tuple[int]) -> T:
        """Reload ``reshape`` from self.tensor."""
        return type(self)(self.tensor.reshape(shape), clone=False)

    def expand(self: T, *sizes: Tuple[int]) -> T:
        """Reload ``expand`` from self.tensor."""
        return type(self)(self.tensor.expand(sizes), clone=False)

    def repeat(self: T, *sizes: Tuple[int]) -> T:
        """Reload ``repeat`` from self.tensor."""
        return type(self)(self.tensor.repeat(sizes), clone=False)

    def transpose(self: T, dim0: int, dim1: int) -> T:
        """Reload ``transpose`` from self.tensor."""
        ndim = self.tensor.dim()
        assert dim0 != -1 and dim0 != ndim - 1
        assert dim1 != -1 and dim1 != ndim - 1
        return type(self)(self.tensor.transpose(dim0, dim1), clone=False)

    def permute(self: T, *dims: Tuple[int]) -> T:
        """Reload ``permute`` from self.tensor."""
        assert dims[-1] == -1 or dims[-1] == self.tensor.dim() - 1
        return type(self)(self.tensor.permute(dims), clone=False)

    def split(self: T,
              split_size_or_sections: Union[int, Sequence[int]],
              dim: int = 0) -> List[T]:
        """Reload ``split`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.split(split_size_or_sections, dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def chunk(self: T, chunks: int, dim: int = 0) -> List[T]:
        """Reload ``chunk`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.chunk(chunks, dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def unbind(self: T, dim: int = 0) -> T:
        """Reload ``unbind`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.unbind(dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def flatten(self: T, start_dim: int = 0, end_dim: int = -2) -> T:
        """Reload ``flatten`` from self.tensor."""
        assert end_dim != -1 and end_dim != self.tensor.dim() - 1
        return type(self)(self.tensor.flatten(start_dim, end_dim), clone=False)

    def squeeze(self: T, dim: Optional[int] = None) -> T:
        """Reload ``squeeze`` from self.tensor."""
        boxes = self.tensor.squeeze() if dim is None else \
            self.tensor.squeeze(dim)
        return type(self)(boxes, clone=False)

    def unsqueeze(self: T, dim: int) -> T:
        """Reload ``unsqueeze`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim()
        return type(self)(self.tensor.unsqueeze(dim), clone=False)

    @classmethod
    def cat(cls: Type[T], box_list: Sequence[T], dim: int = 0) -> T:
        """Cancatenates a box instance list into one single box instance.
        Similar to ``torch.cat``.

        Args:
            box_list (Sequence[T]): A sequence of box instances.
            dim (int): The dimension over which the box are concatenated.
                Defaults to 0.

        Returns:
            T: Concatenated box instance.
        """
        assert isinstance(box_list, Sequence)
        if len(box_list) == 0:
            raise ValueError('box_list should not be a empty list.')

        assert dim != -1 and dim != box_list[0].dim() - 1
        assert all(isinstance(boxes, cls) for boxes in box_list)

        th_box_list = [boxes.tensor for boxes in box_list]
        return cls(torch.cat(th_box_list, dim=dim), clone=False)

    @classmethod
    def stack(cls: Type[T], box_list: Sequence[T], dim: int = 0) -> T:
        """Concatenates a sequence of tensors along a new dimension. Similar to
        ``torch.stack``.

        Args:
            box_list (Sequence[T]): A sequence of box instances.
            dim (int): Dimension to insert. Defaults to 0.

        Returns:
            T: Concatenated box instance.
        """
        assert isinstance(box_list, Sequence)
        if len(box_list) == 0:
            raise ValueError('box_list should not be a empty list.')

        assert dim != -1 and dim != box_list[0].dim()
        assert all(isinstance(boxes, cls) for boxes in box_list)

        th_box_list = [boxes.tensor for boxes in box_list]
        return cls(torch.stack(th_box_list, dim=dim), clone=False)

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

    @abstractmethod
    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        pass

    @abstractmethod
    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        pass

    @abstractmethod
    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip boxes according to the image shape in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        pass

    @abstractmethod
    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        pass

    @abstractmethod
    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        pass

    @abstractmethod
    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        pass

    @abstractmethod
    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        """Resize the box width and height w.r.t scale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.
        """
        pass

    @abstractmethod
    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> BoolTensor:
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.
        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, box_dim),
            the output has shape (m, n).
        """
        pass

    @abstractmethod
    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False) -> BoolTensor:
        """Find inside box points. Boxes dimension must be 2.

        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with boxes
                or not. If True, the length of boxes and ``points`` should be
                the same. Defaults to False.

        Returns:
            BoolTensor: A BoolTensor indicating whether a point is inside
            boxes. Assuming the boxes has shape of (n, box_dim), if
            ``is_aligned`` is False. The index has shape of (m, n). If
            ``is_aligned`` is True, m should be equal to n and the index has
            shape of (m, ).
        """
        pass

    @abstractstaticmethod
    def overlaps(boxes1: 'BaseBoxes',
                 boxes2: 'BaseBoxes',
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6) -> Tensor:
        """Calculate overlap between two set of boxes with their types
        converted to the present box type.

        Args:
            boxes1 (:obj:`BaseBoxes`): BaseBoxes with shape of (m, box_dim)
                or empty.
            boxes2 (:obj:`BaseBoxes`): BaseBoxes with shape of (n, box_dim)
                or empty.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground). Defaults to "iou".
            is_aligned (bool): If True, then m and n must be equal. Defaults
                to False.
            eps (float): A value added to the denominator for numerical
                stability. Defaults to 1e-6.

        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        """
        pass

    @abstractstaticmethod
    def from_instance_masks(masks: MaskType) -> 'BaseBoxes':
        """Create boxes from instance masks.

        Args:
            masks (:obj:`BitmapMasks` or :obj:`PolygonMasks`): BitmapMasks or
                PolygonMasks instance with length of n.

        Returns:
            :obj:`BaseBoxes`: Converted boxes with shape of (n, box_dim).
        """
        pass
