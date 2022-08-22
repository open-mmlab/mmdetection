# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor

from .base_boxes import BaseBoxes

BoxType = Union[np.ndarray, Tensor, BaseBoxes]

box_types: dict = {}
_box_type_to_name: dict = {}
box_converters: dict = {}


def _register_box(name: str, box_type: Type, force: bool = False) -> None:
    """Register a box type.

    Args:
        name (str): The name of box type.
        box_type (type): Box mode class to be registered.
        force (bool): Whether to override an existing class with the same
            name. Defaults to False.
    """
    assert issubclass(box_type, BaseBoxes)
    name = name.lower()

    if not force and (name in box_types or box_type in _box_type_to_name):
        raise KeyError(f'box type {name} has been registered')
    elif name in box_types:
        _box_type = box_types.pop(name)
        _box_type_to_name.pop(_box_type)
    elif box_type in _box_type_to_name:
        _name = _box_type_to_name.pop(box_type)
        box_types.pop(_name)

    box_types[name] = box_type
    _box_type_to_name[box_type] = name


def register_box(name: str,
                 box_type: Type = None,
                 force: bool = False) -> Union[Type, Callable]:
    """Register a box type.

    A record will be added to ``bbox_types``, whose key is the box type name
    and value is the box type itself. Simultaneously, a reverse dictionary
    ``_box_type_to_name`` will be updated. It can be used as a decorator or
    a normal function.

    Args:
        name (str): The name of box type.
        bbox_type (type, Optional): Box type class to be registered.
            Defaults to None.
        force (bool): Whether to override the existing box type with the same
            name. Defaults to False.

    Examples:
        >>> from mmdet.structures.bbox import register_box
        >>> from mmdet.structures.bbox import BaseBoxes

        >>> # as a decorator
        >>> @register_box('hbox')
        >>> class HorizontalBoxes(BaseBoxes):
        >>>     pass

        >>> # as a normal function
        >>> class RotatedBoxes(BaseBoxes):
        >>>     pass
        >>> register_box('rbox', RotatedBoxes)
    """
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')

    # use it as a normal method: register_box(name, box_type=BoxCls)
    if box_type is not None:
        _register_box(name=name, box_type=box_type, force=force)
        return box_type

    # use it as a decorator: @register_box(name)
    def _register(cls):
        _register_box(name=name, box_type=cls, force=force)
        return cls

    return _register


def _register_box_converter(src_type: Union[str, type],
                            dst_type: Union[str, type],
                            converter: Callable,
                            force: bool = False) -> None:
    """Register a box converter.

    Args:
        src_type (str or type): source box type name or class.
        dst_type (str or type): destination box type name or class.
        converter (Callable): Convert function.
        force (bool): Whether to override the existing box type with the same
            name. Defaults to False.
    """
    assert callable(converter)
    src_type_name, _ = get_box_type(src_type)
    dst_type_name, _ = get_box_type(dst_type)

    converter_name = src_type_name + '2' + dst_type_name
    if not force and converter_name in box_converters:
        raise KeyError(f'The box converter from {src_type_name} to '
                       f'{dst_type_name} has been registered.')

    box_converters[converter_name] = converter


def register_box_converter(src_type: Union[str, type],
                           dst_type: Union[str, type],
                           converter: Optional[Callable] = None,
                           force: bool = False) -> Callable:
    """Register a box converter.

    A record will be added to ``box_converter``, whose key is
    '{src_type_name}2{dst_type_name}' and value is the convert function.
    It can be used as a decorator or a normal function.

    Args:
        src_type (str or type): source box type name or class.
        dst_type (str or type): destination box type name or class.
        converter (Callable): Convert function. Defaults to None.
        force (bool): Whether to override the existing box type with the same
            name. Defaults to False.

    Examples:
        >>> from mmdet.structures.bbox import register_box_converter
        >>> # as a decorator
        >>> @register_box_converter('hbox', 'rbox')
        >>> def converter_A(boxes):
        >>>     pass

        >>> # as a normal function
        >>> def converter_B(boxes):
        >>>     pass
        >>> register_box_converter('rbox', 'hbox', converter_B)
    """
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')

    # use it as a normal method:
    # register_box_converter(src_type, dst_type, converter=Func)
    if converter is not None:
        _register_box_converter(
            src_type=src_type,
            dst_type=dst_type,
            converter=converter,
            force=force)
        return converter

    # use it as a decorator: @register_box_converter(name)
    def _register(func):
        _register_box_converter(
            src_type=src_type, dst_type=dst_type, converter=func, force=force)
        return func

    return _register


def get_box_type(box_type: Union[str, type]) -> Tuple[str, type]:
    """get both box type name and class.

    Args:
        box_type (str or type): Single box type name or class.

    Returns:
        Tuple[str, type]: A tuple of box type name and class.
    """
    if isinstance(box_type, str):
        type_name = box_type.lower()
        assert type_name in box_types, \
            f"Box type {type_name} hasn't been registered in box_types."
        type_cls = box_types[type_name]
    elif issubclass(box_type, BaseBoxes):
        assert box_type in _box_type_to_name, \
            f"Box type {box_type} hasn't been registered in box_types."
        type_name = _box_type_to_name[box_type]
        type_cls = box_type
    else:
        raise KeyError('box_type must be a str or class inheriting from '
                       f'BaseBoxes, but got {type(box_type)}.')
    return type_name, type_cls


def convert_box_type(boxes: BoxType,
                     *,
                     src_type: Union[str, type] = None,
                     dst_type: Union[str, type] = None) -> BoxType:
    """Convert boxes from source type to destination type.

    If ``boxes`` is a instance of BaseBoxes, the ``src_type`` will be set
    as the type of ``boxes``.

    Args:
        boxes (np.ndarray or Tensor or :obj:`BaseBoxes`): boxes need to
            convert.
        src_type (str or type, Optional): source box type. Defaults to None.
        dst_type (str or type, Optional): destination box type. Defaults to
            None.

    Returns:
        Union[np.ndarray, Tensor, :obj:`BaseBoxes`]: Converted boxes. It's type
        is consistent with the input's type.
    """
    assert dst_type is not None
    dst_type_name, dst_type_cls = get_box_type(dst_type)

    is_box_cls = False
    is_numpy = False
    if isinstance(boxes, BaseBoxes):
        src_type_name, _ = get_box_type(type(boxes))
        is_box_cls = True
    elif isinstance(boxes, (Tensor, np.ndarray)):
        assert src_type is not None
        src_type_name, _ = get_box_type(src_type)
        if isinstance(boxes, np.ndarray):
            is_numpy = True
    else:
        raise TypeError('boxes must be a instance of BaseBoxes, Tensor or '
                        f'ndarray, but get {type(boxes)}.')

    if src_type_name == dst_type_name:
        return boxes

    converter_name = src_type_name + '2' + dst_type_name
    assert converter_name in box_converters, \
        "Convert function hasn't been registered in box_converters."
    converter = box_converters[converter_name]

    if is_box_cls:
        boxes = converter(boxes.tensor)
        return dst_type_cls(boxes)
    elif is_numpy:
        boxes = converter(torch.from_numpy(boxes))
        return boxes.numpy()
    else:
        return converter(boxes)


def autocast_box_type(dst_box_type='hbox') -> Callable:
    """A decorator which automatically casts results['gt_bboxes'] to the
    destination box type.

    It commenly used in mmdet.datasets.transforms to make the transforms up-
    compatible with the np.ndarray type of results['gt_bboxes'].

    The speed of processing of np.ndarray and BaseBoxes data are the same:

    - np.ndarray: 0.0509 img/s
    - BaseBoxes: 0.0551 img/s

    Args:
        dst_box_type (str): Destination box type.
    """
    _, box_type_cls = get_box_type(dst_box_type)

    def decorator(func: Callable) -> Callable:

        def wrapper(self, results: dict, *args, **kwargs) -> dict:
            if ('gt_bboxes' not in results
                    or isinstance(results['gt_bboxes'], BaseBoxes)):
                return func(self, results)
            elif isinstance(results['gt_bboxes'], np.ndarray):
                results['gt_bboxes'] = box_type_cls(
                    results['gt_bboxes'], clone=False)
                if 'mix_results' in results:
                    for res in results['mix_results']:
                        if isinstance(res['gt_bboxes'], np.ndarray):
                            res['gt_bboxes'] = box_type_cls(
                                res['gt_bboxes'], clone=False)

                _results = func(self, results, *args, **kwargs)

                # In some cases, the function will process gt_bboxes in-place
                # Simultaneously convert inputting and outputting gt_bboxes
                # back to np.ndarray
                if isinstance(_results, dict) and 'gt_bboxes' in _results:
                    if isinstance(_results['gt_bboxes'], BaseBoxes):
                        _results['gt_bboxes'] = _results['gt_bboxes'].numpy()
                if isinstance(results['gt_bboxes'], BaseBoxes):
                    results['gt_bboxes'] = results['gt_bboxes'].numpy()
                return _results
            else:
                raise TypeError(
                    "auto_box_type requires results['gt_bboxes'] to "
                    'be BaseBoxes or np.ndarray, but got '
                    f"{type(results['gt_bboxes'])}")

        return wrapper

    return decorator
