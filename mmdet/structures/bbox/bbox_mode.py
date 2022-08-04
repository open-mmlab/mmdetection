# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor

from .base_bbox import BaseBoxes

BoxType = Union[np.ndarray, Tensor, BaseBoxes]

bbox_modes: dict = {}
bbox_mode_converters: dict = {}
_bbox_mode_to_name: dict = {mode: name for name, mode in bbox_modes.items()}


def _register_bbox_mode(name: str,
                        bbox_mode: Type,
                        force: bool = False) -> None:
    """Register a box mode.

    Args:
        name (str): The name of box mode.
        bbox_mode (type): Box mode class to be registered.
        force (bool): Whether to override an existing class with the same
            name. Defaults to False.
    """
    assert issubclass(bbox_mode, BaseBoxes)
    name = name.lower()

    if not force and (name in bbox_modes or bbox_mode in _bbox_mode_to_name):
        raise KeyError('name or bbox_mode has been register')
    elif name in bbox_modes:
        _bbox_mode = bbox_modes.pop(name)
        _bbox_mode_to_name.pop(_bbox_mode)
    elif bbox_mode in _bbox_mode_to_name:
        _name = _bbox_mode_to_name.pop(bbox_mode)
        bbox_modes.pop(_name)

    bbox_modes[name] = bbox_mode
    _bbox_mode_to_name[bbox_mode] = name


def register_bbox_mode(name: str,
                       bbox_mode: Type = None,
                       force: bool = False) -> Union[Type, Callable]:
    """Register a box mode.

    A record will be added to ``bbox_modes``, whose key is the box mode
    name and value is the box mode class itself. Simultaneously, a reverse
    dictionary ``_bbox_mode_to_name`` will also update. It can be used as
    a decorator or a normal function.

    Args:
        name (str): The name of box mode.
        bbox_mode (type, Optional): Box mode class to be registered.
            Defaults to None.
        force (bool): Whether to override an existing class with the same
            name. Defaults to False.

    Examples:
        >>> from mmdet.structures.bbox import register_bbox_mode
        >>> from mmdet.structures.bbox import BaseBoxes

        >>> # as a decorator
        >>> @register_bbox_mode('hbox')
        >>> class HoriBoxes(BaseBoxes):
        >>>     pass

        >>> # as a normal function
        >>> class RotInstanceBoxes(BaseBoxes):
        >>>     pass
        >>> register_bbox_mode('rbox', RotInstanceBoxes)
    """
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')

    # use it as a normal method: register_bbox_mode(name, bbox_mode=BBoxCls)
    if bbox_mode is not None:
        _register_bbox_mode(name=name, bbox_mode=bbox_mode, force=force)
        return bbox_mode

    # use it as a decorator: @register_bbox_mode(name)
    def _register(bbox_cls):
        _register_bbox_mode(name=name, bbox_mode=bbox_cls, force=force)
        return bbox_cls

    return _register


def _register_bbox_mode_converter(src_mode: str,
                                  dst_mode: str,
                                  converter: Callable,
                                  force: bool = False) -> None:
    """Register a box mode converter.

    Args:
        src_mode (str): The name of source box mode.
        dst_mode (str): The name of destination box mode.
        converter (Callable): Convert function.
        force (bool): Whether to override an existing class with the same
            name. Defaults to False.
    """
    assert callable(converter)
    assert isinstance(src_mode, str)
    assert isinstance(dst_mode, str)
    src_mode, dst_mode = src_mode.lower(), dst_mode.lower()
    assert src_mode in bbox_modes and dst_mode in bbox_modes, \
        'Boxes mode should be register'

    converter_name = src_mode + '2' + dst_mode
    if not force and converter_name in bbox_mode_converters:
        raise KeyError('convert function has been registered.')

    bbox_mode_converters[converter_name] = converter


def register_bbox_mode_converter(src_mode: str,
                                 dst_mode: str,
                                 converter: Optional[Callable] = None,
                                 force: bool = False) -> Callable:
    """Register a box mode converter.

    A record will be added to ``bbox_mode_converters``, whose key is
    '{src_mode}2{dst_mode}' and value is the convert function.  It can be
    used as a decorator or a normal function.

    Args:
        src_mode (str): The name of source box mode.
        dst_mode (str): The name of destination box mode.
        converter (Callable, Optional): Convert function. Defaults to None.
        force (bool): Whether to override an existing class with the same
            name. Defaults to False.

    Examples:
        >>> from mmdet.structures.bbox import register_mode_converter
        >>> # as a decorator
        >>> @register_mode_converter('hbox', 'rbox')
        >>> def converter_A(bboxes):
        >>>     pass

        >>> # as a normal function
        >>> def converter_B(bboxes):
        >>>     pass
        >>> register_mode_converter('rbox', 'hbox', converter_B)
    """
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')

    # use it as a normal method:
    # register_convert_func(src_mode, dst_mode, convert_func=Func)
    if converter is not None:
        _register_bbox_mode_converter(
            src_mode=src_mode,
            dst_mode=dst_mode,
            converter=converter,
            force=force)
        return converter

    # use it as a decorator: @register_bbox_mode(name)
    def _register(func):
        _register_bbox_mode_converter(
            src_mode=src_mode, dst_mode=dst_mode, converter=func, force=force)
        return func

    return _register


def get_bbox_mode(mode: Union[str, type]) -> Tuple[str, type]:
    """get box mode name and class.

    Args:
        mode (str or type): Single box mode name or class.

    Returns:
        Union[str, type]: A tuple of box mode name and class.
    """
    if isinstance(mode, str):
        mode_name = mode.lower()
        assert mode_name in bbox_modes, \
            f"Mode {mode_name} hasn't been registered in bbox_modes."
        mode_cls = bbox_modes[mode_name]
    elif issubclass(mode, BaseBoxes):
        assert mode in _bbox_mode_to_name, \
            f"Mode {mode} hasn't been registered in bbox_modes."
        mode_name = _bbox_mode_to_name[mode]
        mode_cls = mode
    else:
        raise KeyError('Expect str or BaseBoxes subclass inputs, '
                       f'but get {type(mode)}.')
    return mode_name, mode_cls


def convert_bbox_mode(bboxes: BoxType,
                      *,
                      src_mode: Union[str, type] = None,
                      dst_mode: Union[str, type] = None) -> BoxType:
    """Convert bboxes from source mode to destination mode.

    If bboxes is a instance of BaseBoxes, the src_mode will be set
    as the mode of bboxes.

    Args:
        bboxes (np.ndarray or Tensor or BaseBoxes): boxes need to
            convert.
        src_mode (str or type, Optional): source box mode. Defaults to None.
        dst_mode (str or type, Optional): destination box mode. Defaults to
            None.
    """
    assert dst_mode is not None
    dst_mode_name, dst_mode_cls = get_bbox_mode(dst_mode)

    is_bbox_cls = False
    is_numpy = False
    if isinstance(bboxes, BaseBoxes):
        src_mode_name, _ = get_bbox_mode(type(bboxes))
        is_bbox_cls = True
    elif isinstance(bboxes, (Tensor, np.ndarray)):
        assert src_mode is not None
        src_mode_name, _ = get_bbox_mode(src_mode)
        if isinstance(bboxes, np.ndarray):
            is_numpy = True
    else:
        raise TypeError('bboxes needs to be BaseBoxes, Tensor or '
                        f'ndarray, but get {type(bboxes)}.')

    if src_mode_name == dst_mode_name:
        return bboxes

    converter_name = src_mode_name + '2' + dst_mode_name
    assert converter_name in bbox_mode_converters, \
        "Convert function hasn't been registered in bbox_mode_converters."
    converter = bbox_mode_converters[converter_name]

    if is_bbox_cls:
        bboxes = converter(bboxes.tensor)
        return dst_mode_cls(bboxes)
    elif is_numpy:
        bboxes = converter(torch.from_numpy(bboxes))
        return bboxes.numpy()
    else:
        return converter(bboxes)
