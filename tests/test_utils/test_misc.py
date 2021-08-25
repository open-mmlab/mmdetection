# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet.core.bbox import distance2bbox
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


def test_distance2bbox():
    point = torch.Tensor([[74., 61.], [-29., 106.], [138., 61.], [29., 170.]])

    distance = torch.Tensor([[0., 0, 1., 1.], [1., 2., 10., 6.],
                             [22., -29., 138., 61.], [54., -29., 170., 61.]])
    expected_decode_bboxes = torch.Tensor([[74., 61., 75., 62.],
                                           [0., 104., 0., 112.],
                                           [100., 90., 100., 120.],
                                           [0., 120., 100., 120.]])
    out_bbox = distance2bbox(point, distance, max_shape=(120, 100))
    assert expected_decode_bboxes.allclose(out_bbox)
    out = distance2bbox(point, distance, max_shape=torch.Tensor((120, 100)))
    assert expected_decode_bboxes.allclose(out)

    batch_point = point.unsqueeze(0).repeat(2, 1, 1)
    batch_distance = distance.unsqueeze(0).repeat(2, 1, 1)
    batch_out = distance2bbox(
        batch_point, batch_distance, max_shape=(120, 100))[0]
    assert out.allclose(batch_out)
    batch_out = distance2bbox(
        batch_point, batch_distance, max_shape=[(120, 100), (120, 100)])[0]
    assert out.allclose(batch_out)

    batch_out = distance2bbox(point, batch_distance, max_shape=(120, 100))[0]
    assert out.allclose(batch_out)

    # test max_shape is not equal to batch
    with pytest.raises(AssertionError):
        distance2bbox(
            batch_point,
            batch_distance,
            max_shape=[(120, 100), (120, 100), (32, 32)])

    rois = torch.zeros((0, 4))
    deltas = torch.zeros((0, 4))
    out = distance2bbox(rois, deltas, max_shape=(120, 100))
    assert rois.shape == out.shape

    rois = torch.zeros((2, 0, 4))
    deltas = torch.zeros((2, 0, 4))
    out = distance2bbox(rois, deltas, max_shape=(120, 100))
    assert rois.shape == out.shape
