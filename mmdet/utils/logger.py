# Copyright (c) OpenMMLab. All rights reserved.
import inspect

from mmengine.logging import print_log


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

    caller = get_caller_name()
    print_log(
        f'image shape: height={height}, width={width} in {caller}',
        logger='current')

    return True
