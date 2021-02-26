import numpy as np
import pytest
import torch

from mmdet.core.mask.structures import BitmapMasks, PolygonMasks
from mmdet.core.utils import mask2ndarray


def dummy_raw_polygon_masks(size):
    """
    Args:
        size (tuple): expected shape of dummy masks, (N, H, W)

    Return:
        list[list[ndarray]]: dummy mask
    """
    num_obj, heigt, width = size
    polygons = []
    for _ in range(num_obj):
        num_points = np.random.randint(5) * 2 + 6
        polygons.append([np.random.uniform(0, min(heigt, width), num_points)])
    return polygons


def test_mask2ndarray():
    raw_masks = np.ones((3, 28, 28))
    bitmap_mask = BitmapMasks(raw_masks, 28, 28)
    output_mask = mask2ndarray(bitmap_mask)
    assert np.allclose(raw_masks, output_mask)

    raw_masks = dummy_raw_polygon_masks((3, 28, 28))
    polygon_masks = PolygonMasks(raw_masks, 28, 28)
    output_mask = mask2ndarray(polygon_masks)
    assert output_mask.shape == (3, 28, 28)

    raw_masks = np.ones((3, 28, 28))
    output_mask = mask2ndarray(raw_masks)
    assert np.allclose(raw_masks, output_mask)

    raw_masks = torch.ones((3, 28, 28))
    output_mask = mask2ndarray(raw_masks)
    assert np.allclose(raw_masks, output_mask)

    # test unsupported type
    raw_masks = []
    with pytest.raises(TypeError):
        output_mask = mask2ndarray(raw_masks)
