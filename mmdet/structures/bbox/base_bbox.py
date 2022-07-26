# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractclassmethod
from inspect import signature
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import Tensor


class BaseBBoxes(metaclass=ABCMeta):
    """Base class for bounding boxes.

    Args:
        bboxes(Tensor or np.ndarray): The bbox data.
        src_submode (str, optional): The submode of bboxes. BaseBBoxes will
            according to src_submode to convert bboxes into regular format.
            Defaults to None.
        device(str or torch.device): device of bboxes.
        dtype(torch.dtype): data type of bboxes.
    """

    _bbox_dim: int = 0
    SUBMODE_CONVERTERS: dict = None

    def __init__(self,
                 bboxes: Union[Tensor, np.ndarray],
                 src_submode: Optional[str] = None,
                 device: Union[str, torch.device] = None,
                 dtype: torch.dtype = None) -> None:
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes)
        assert isinstance(bboxes, Tensor)

        if device is not None or dtype is not None:
            bboxes = bboxes.to(dtype=dtype, device=device)
        if src_submode is not None:
            assert src_submode in self.SUBMODE_CONVERTERS, \
                f"{src_submode} hasn't been registered in {type(self)}."
            converter = self.SUBMODE_CONVERTERS[src_submode]
            bboxes = converter(bboxes=bboxes, direction='from_submode')

        assert bboxes.dim() >= 2
        assert bboxes.size(-1) == self._bbox_dim
        self._tensor = bboxes

    @classmethod
    def _register_submode(cls,
                          name: str,
                          converter: Callable,
                          force: bool = False) -> None:
        name = name.lower()
        assert callable(converter), 'Converter should be a function.'
        if not force and name in cls.SUBMODE_CONVERTERS:
            raise KeyError(f'{name} has been registered in {cls}.')

        cvt_sig = signature(converter)
        cvt_args = [p.name for p in cvt_sig.parameters.values()]
        assert 'bboxes' in cvt_args and 'direction' in cvt_args, \
            'Converter must have `bboxes` and `direction` arguments.'

        cls.SUBMODE_CONVERTERS[name] = converter

    @classmethod
    def register_submode(cls, name, converter=None, force=False):
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # use it as a normal method: x.register_submode(name, converter=Func)
        if converter is not None:
            cls._register_submode(name=name, converter=converter, force=force)
            return converter

        # use it as a decorator: @x.register_submode(name)
        def _register(func):
            cls._register_submode(name=name, converter=func, force=force)
            return func

        return _register

    def to_submode(self, dst_submode):
        dst_submode = dst_submode.lower()
        assert dst_submode in self.SUBMODE_CONVERTERS, \
            f"{dst_submode} hasn't been registered in {type(self)}."
        converter = self.SUBMODE_CONVERTERS[dst_submode]
        return converter(bboxes=self._tensor, direction='to_submode')

    def to_mode(self, dst_mode):
        from .bbox_mode import convert_bbox_mode
        return convert_bbox_mode(self, dst_mode=dst_mode)

    @property
    def bbox_dim(self):
        return self._bbox_dim

    @property
    def tensor(self):
        return self._tensor

    def create_empty_bboxes(self, dtype=None, device=None):
        empty_bboxes = self._tensor.new_zeros((0, self._bbox_dim),
                                              dtype=dtype,
                                              device=device)
        return type(self)(empty_bboxes)

    def create_fake_bboxes(self, *sizes, dtype=None, device=None):
        fake_bboxes = self._tensor.new_zeros(
            sizes + (self._bbox_dim, ), dtype=dtype, device=device)
        return type(self)(fake_bboxes)

    def __getitem__(self, index):
        bboxes = self._tensor
        if isinstance(index, tuple):
            assert len(index) < bboxes.dim()
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        bboxes = bboxes[index]
        return type(self)(bboxes)

    def __setitem__(self, index, values):
        if isinstance(values, BaseBBoxes):
            assert isinstance(values, type(self)), \
                f'Cannot set {type(values)} into {type(self)}'
            values = values._tensor

        if isinstance(index, tuple):
            assert len(index) < self._tensor.dim()
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        self._tensor[index] = values

    def __len__(self):
        return self.tensor.size(0)

    def __iter__(self):
        bboxes = self._tensor
        if bboxes.dim() > 2:
            return (type(self)(bboxes[i]) for i in range(bboxes.size(0)))
        else:
            return (type(self)(bboxes[[i]]) for i in range(bboxes.size(0)))

    def __repr__(self):
        return self.__class__.__name__ + '(\n' + str(self.tensor) + ')'

    def new_tensor(self, *args, **kwargs):
        return self._tensor.new_tensor(*args, **kwargs)

    def new_full(self, *args, **kwargs):
        return self._tensor.new_full(*args, **kwargs)

    def new_empty(self, *args, **kwargs):
        return self._tensor.new_empty(*args, **kwargs)

    def new_ones(self, *args, **kwargs):
        return self._tensor.new_ones(*args, **kwargs)

    def new_zeros(self, *args, **kwargs):
        return self._tensor.new_zeros(*args, **kwargs)

    def size(self, dim=None):
        return self._tensor.size(dim)

    def dim(self):
        return self._tensor.dim()

    @property
    def device(self):
        return self._tensor.device

    @property
    def dtype(self):
        return self._tensor.dtype

    def to(self, *args, **kwargs):
        return type(self)(self._tensor.to(*args, **kwargs))

    def cpu(self):
        return type(self)(self._tensor.cpu())

    def cuda(self, *args, **kwargs):
        return type(self)(self._tensor.cuda(*args, **kwargs))

    def clone(self):
        return type(self)(self._tensor.clone())

    def detach(self):
        return type(self)(self._tensor.detach())

    def view(self, *shape):
        return type(self)(self._tensor.view(shape))

    def expand(self, *sizes):
        return type(self)(self._tensor.expand(sizes))

    def repeat(self, *sizes):
        return type(self)(self._tensor.repeat(sizes))

    def permute(self, *dims):
        assert dims[-1] in (-1, self._tensor.dim() - 1)
        return type(self)(self._tensor.permute(dims))

    def split(self, split_size_or_sections, dim=0):
        assert dim not in (-1, self._tensor.dim() - 1)
        bboxes_list = self._tensor.split(split_size_or_sections, dim=dim)
        return [type(self)(bboxes) for bboxes in bboxes_list]

    def chunk(self, chunks, dim=0):
        assert dim not in (-1, self._tensor.dim() - 1)
        bboxes_list = self._tensor.chunk(chunks, dim=dim)
        return [type(self)(bboxes) for bboxes in bboxes_list]

    def flatten(self, start_dim=0, end_dim=-2):
        assert end_dim not in (-1, self._tensor.dim() - 1)
        return type(self)(self._tensor.flatten(start_dim, end_dim))

    @abstractclassmethod
    @property
    def centers(self):
        pass

    @abstractclassmethod
    @property
    def areas(self):
        pass

    @abstractclassmethod
    @property
    def widths(self):
        pass

    @abstractclassmethod
    @property
    def heights(self):
        pass

    @abstractclassmethod
    def flip(self, direction, img_shape):
        pass

    @abstractclassmethod
    def translate(self, distances):
        pass

    @abstractclassmethod
    def clip(self, border):
        pass

    @abstractclassmethod
    def rotate(self, center, angle):
        pass

    @abstractclassmethod
    def project(self, mat, keep_type=True):
        # mmdet/structures/bbox/transforms/bbox_project
        pass

    @abstractclassmethod
    def rescale(self, rescale_factor):
        pass

    @abstractclassmethod
    def rescale_size(self, rescale_factor):
        pass

    @abstractclassmethod
    def is_bboxes_inside(self, border):
        pass

    @abstractclassmethod
    def is_points_inside(self, points, is_aligned=False):
        pass
