# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

import mmdet


def palette_val(palette):
    """Convert palette to matplotlib palette.

    Args:
        palette List[tuple]: A list of color tuples.

    Returns:
        List[tuple[float]]: A list of RGB matplotlib color tuples.
    """
    new_palette = []
    for color in palette:
        color = [c / 255 for c in color]
        new_palette.append(tuple(color))
    return new_palette


def get_palette(palette, num_classes=None):
    """Get palette from various inputs.

    Args:
        palette (list[tuple]/str/tuple/:obj:`Color`): palette inputs

    Returns:
        list[tuple[int]]: A list of color tuples.
    """
    if isinstance(palette, list):
        return palette
    elif isinstance(palette, tuple):
        assert isinstance(num_classes, int)
        return [palette] * num_classes
    elif palette == 'coco':
        return mmdet.datasets.CocoDataset.PALETTE
    elif palette == 'voc':
        return mmdet.datasets.VOCDataset.PALETTE
    elif palette == 'citys':
        return mmdet.datasets.CityscapesDataset.PALETTE
    elif palette == 'random' or palette is None:
        assert isinstance(num_classes, int)
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        return [tuple(c) for c in palette]
    elif mmcv.is_str(palette):
        assert isinstance(num_classes, int)
        return [mmcv.color_val(palette)[::-1]] * num_classes
    else:
        raise TypeError(f'Invalid type for palette: {type(palette)}')
