# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch import Tensor

from .base_bbox import BaseBBoxes

bbox_modes = {}
convert_functions = {}

_bbox_mode_to_name = {mode: name for name, mode in bbox_modes.items()}


def _register_bbox_mode(name, bbox_mode, force=False):
    assert isinstance(bbox_mode, BaseBBoxes)
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


def register_bbox_mode(name, bbox_mode=None, force=False):
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


def _register_convert_func(src_mode, dst_mode, convert_func, force=False):
    assert callable(convert_func)
    assert isinstance(src_mode, str)
    assert isinstance(dst_mode, str)
    src_mode, dst_mode = src_mode.lower(), dst_mode.lower()
    assert src_mode in bbox_modes and dst_mode in bbox_modes, \
        'Boxes mode should be register'

    convert_func_name = src_mode + '2' + dst_mode
    if not force and convert_func_name in convert_functions:
        raise KeyError('convert function has been registered.')

    convert_functions[convert_func_name] = convert_func


def register_convert_func(src_mode, dst_mode, convert_func=None, force=False):
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')

    # use it as a normal method:
    # register_convert_func(src_mode, dst_mode, convert_func=Func)
    if convert_func is not None:
        _register_convert_func(
            src_mode=src_mode,
            dst_mode=dst_mode,
            convert_func=convert_func,
            force=force)
        return convert_func

    # use it as a decorator: @register_bbox_mode(name)
    def _register(func):
        _register_convert_func(
            src_mode=src_mode,
            dst_mode=dst_mode,
            convert_func=func,
            force=force)
        return func

    return _register


def convert_bbox_mode(bboxes, *, src_mode=None, dst_mode=None):
    assert dst_mode is not None
    dst_mode = dst_mode.lower()

    is_bbox_cls = False
    is_numpy = False
    if isinstance(bboxes, BaseBBoxes):
        src_mode = _bbox_mode_to_name[type(bboxes)]
        is_bbox_cls = True
    elif isinstance(bboxes, (Tensor, np.ndarray)):
        assert isinstance(src_mode, str)
        src_mode = src_mode.lower()
        if isinstance(bboxes, np.ndarray):
            is_numpy = True
    else:
        raise TypeError('only accept BBoxes, Tensor or ndarray')

    if src_mode == dst_mode:
        return bboxes

    func_name = src_mode + '2' + dst_mode
    assert func_name in convert_functions, \
        "Convert function hasn't been registered"
    convert_func = convert_functions[func_name]

    if is_bbox_cls:
        bboxes = convert_func(bboxes.tensor)
        dst_bbox_cls = bbox_modes[dst_mode]
        return dst_bbox_cls(bboxes)
    elif is_numpy:
        bboxes = convert_func(torch.from_numpy(bboxes))
        return bboxes.numpy()
    else:
        return convert_func(bboxes)
