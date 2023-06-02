# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import mmcv
import numpy as np
from mmengine.utils import is_str


def palette_val(palette: List[tuple]) -> List[tuple]:
    """Convert palette to matplotlib palette.

    Args:
        palette (List[tuple]): A list of color tuples.

    Returns:
        List[tuple[float]]: A list of RGB matplotlib color tuples.
    """
    new_palette = []
    for color in palette:
        color = [c / 255 for c in color]
        new_palette.append(tuple(color))
    return new_palette


def get_palette(palette: Union[List[tuple], str, tuple],
                num_classes: int) -> List[Tuple[int]]:
    """Get palette from various inputs.

    Args:
        palette (list[tuple] | str | tuple): palette inputs.
        num_classes (int): the number of classes.

    Returns:
        list[tuple[int]]: A list of color tuples.
    """
    assert isinstance(num_classes, int)

    if isinstance(palette, list):
        dataset_palette = palette
    elif isinstance(palette, tuple):
        dataset_palette = [palette] * num_classes
    elif palette == 'random' or palette is None:
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    elif palette == 'coco':
        from mmdet.datasets import CocoDataset, CocoPanopticDataset
        dataset_palette = CocoDataset.METAINFO['palette']
        if len(dataset_palette) < num_classes:
            dataset_palette = CocoPanopticDataset.METAINFO['palette']
    elif palette == 'citys':
        from mmdet.datasets import CityscapesDataset
        dataset_palette = CityscapesDataset.METAINFO['palette']
    elif palette == 'voc':
        from mmdet.datasets import VOCDataset
        dataset_palette = VOCDataset.METAINFO['palette']
    elif is_str(palette):
        dataset_palette = [mmcv.color_val(palette)[::-1]] * num_classes
    else:
        raise TypeError(f'Invalid type for palette: {type(palette)}')

    assert len(dataset_palette) >= num_classes, \
        'The length of palette should not be less than `num_classes`.'
    return dataset_palette


def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def jitter_color(color: tuple) -> tuple:
    """Randomly jitter the given color in order to better distinguish instances
    with the same class.

    Args:
        color (tuple): The RGB color tuple. Each value is between [0, 255].

    Returns:
        tuple: The jittered color tuple.
    """
    jitter = np.random.rand(3)
    jitter = (jitter / np.linalg.norm(jitter) - 0.5) * 0.5 * 255
    color = np.clip(jitter + color, 0, 255).astype(np.uint8)
    return tuple(color)
