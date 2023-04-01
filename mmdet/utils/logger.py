# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger


def get_caller_name():
    """Get name of caller method."""
    # this_func_frame = inspect.stack()[0][0]  # i.e., get_caller_name
    # callee_frame = inspect.stack()[1][0]  # e.g., log_img_scale
    caller_frame = inspect.stack()[2][0]  # e.g., caller of log_img_scale
    caller_method = caller_frame.f_code.co_name
    try:
        caller_class = caller_frame.f_locals['self'].__class__.__name__
        return f'{caller_class}.{caller_method}'
    except KeyError:  # caller is a function
        return caller_method


def log_img_scale(img_scale, shape_order='hw', skip_square=False):
    """Log image size.

    Args:
        img_scale (tuple): Image size to be logged.
        shape_order (str, optional): The order of image shape.
            'hw' for (height, width) and 'wh' for (width, height).
            Defaults to 'hw'.
        skip_square (bool, optional): Whether to skip logging for square
            img_scale. Defaults to False.

    Returns:
        bool: Whether to have done logging.
    """
    if shape_order == 'hw':
        height, width = img_scale
    elif shape_order == 'wh':
        width, height = img_scale
    else:
        raise ValueError(f'Invalid shape_order {shape_order}.')

    if skip_square and (height == width):
        return False

    logger = get_root_logger()
    caller = get_caller_name()
    logger.info(f'image shape: height={height}, width={width} in {caller}')

    return True
