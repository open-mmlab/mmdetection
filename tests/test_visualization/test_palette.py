# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmdet.datasets import CocoDataset
from mmdet.visualization import get_palette, jitter_color, palette_val


def test_palette():
    assert palette_val([(1, 2, 3)])[0] == (1 / 255, 2 / 255, 3 / 255)

    # test list
    palette = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    palette_ = get_palette(palette, 3)
    for color, color_ in zip(palette, palette_):
        assert color == color_

    # test tuple
    palette = get_palette((1, 2, 3), 3)
    assert len(palette) == 3
    for color in palette:
        assert color == (1, 2, 3)

    # test color str
    palette = get_palette('red', 3)
    assert len(palette) == 3
    for color in palette:
        assert color == (255, 0, 0)

    # test dataset str
    palette = get_palette('coco', len(CocoDataset.METAINFO['classes']))
    assert len(palette) == len(CocoDataset.METAINFO['classes'])
    assert palette[0] == (220, 20, 60)

    # TODO: Awaiting refactoring
    # palette = get_palette('coco', len(CocoPanopticDataset.METAINFO['CLASSES'])) # noqa
    # assert len(palette) == len(CocoPanopticDataset.METAINFO['CLASSES'])
    # assert palette[-1] == (250, 141, 255)
    # palette = get_palette('voc', len(VOCDataset.METAINFO['CLASSES']))
    # assert len(palette) == len(VOCDataset.METAINFO['CLASSES'])
    # assert palette[0] == (106, 0, 228)
    # palette = get_palette('citys', len(CityscapesDataset.METAINFO['CLASSES'])) # noqa
    # assert len(palette) == len(CityscapesDataset.METAINFO['CLASSES'])
    # assert palette[0] == (220, 20, 60)

    # test random
    palette1 = get_palette('random', 3)
    palette2 = get_palette(None, 3)
    for color1, color2 in zip(palette1, palette2):
        assert isinstance(color1, tuple)
        assert isinstance(color2, tuple)
        assert color1 == color2


def test_jitter_color():
    color = tuple(np.random.randint(0, 255, 3, np.uint8))
    jittered_color = jitter_color(color)
    for c in jittered_color:
        assert 0 <= c <= 255
