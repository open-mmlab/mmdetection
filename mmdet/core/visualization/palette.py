from enum import Enum

import mmcv
import numpy as np

import mmdet


class Color(Enum):
    """An enum that defines common colors.

    Contains red, green ,blue, cyan, yellow, magenta, white and black.
    """
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    cyan = (0, 255, 255)
    yellow = (255, 255, 0)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


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
    elif mmcv.is_str(palette):
        if palette == 'coco':
            palette = mmdet.datasets.CocoDataset.PALETTE
        elif palette == 'voc':
            palette = mmdet.datasets.VOCDataset.PALETTE
        elif palette == 'citys':
            palette = mmdet.datasets.CityscapesDataset.PALETTE
        elif palette == 'random' or palette is None:
            assert isinstance(num_classes, int)
            state = np.random.get_state()
            # random color
            np.random.seed(42)
            palette = np.random.randint(0, 256, size=(num_classes, 3))
            palette = [tuple(c) for c in palette]
            np.random.set_state(state)
        else:
            assert isinstance(num_classes, int)
            return [Color[palette].value] * num_classes
    elif isinstance(palette, Color):
        assert isinstance(num_classes, int)
        return [palette.value] * num_classes
    elif isinstance(palette, tuple):
        assert isinstance(num_classes, int)
        return [palette] * num_classes
    return palette
