# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet.core.bbox import distance2bbox
from mmdet.core.mask.structures import BitmapMasks, PolygonMasks
from mmdet.core.utils import (center_of_mass, filter_scores_and_topk,
                              flip_tensor, mask2ndarray, select_single_mlvl)


def dummy_raw_polygon_masks(size):
    """
    Args:
        size (tuple): expected shape of dummy masks, (N, H, W)

    Return:
        list[list[ndarray]]: dummy mask
    """
    num_obj, height, width = size
    polygons = []
    for _ in range(num_obj):
        num_points = np.random.randint(5) * 2 + 6
        polygons.append([np.random.uniform(0, min(height, width), num_points)])
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


@pytest.mark.parametrize('mask', [
    torch.ones((28, 28)),
    torch.zeros((28, 28)),
    torch.rand(28, 28) > 0.5,
    torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
])
def test_center_of_mass(mask):
    center_h, center_w = center_of_mass(mask)
    if mask.shape[0] == 4:
        assert center_h == 1.5
        assert center_w == 1.5
    assert isinstance(center_h, torch.Tensor) \
           and isinstance(center_w, torch.Tensor)
    assert 0 <= center_h <= 28 \
           and 0 <= center_w <= 28


def test_flip_tensor():
    img = np.random.random((1, 3, 10, 10))
    src_tensor = torch.from_numpy(img)

    # test flip_direction parameter error
    with pytest.raises(AssertionError):
        flip_tensor(src_tensor, 'flip')

    # test tensor dimension
    with pytest.raises(AssertionError):
        flip_tensor(src_tensor[0], 'vertical')

    hfilp_tensor = flip_tensor(src_tensor, 'horizontal')
    expected_hflip_tensor = torch.from_numpy(img[..., ::-1, :].copy())
    expected_hflip_tensor.allclose(hfilp_tensor)

    vfilp_tensor = flip_tensor(src_tensor, 'vertical')
    expected_vflip_tensor = torch.from_numpy(img[..., ::-1].copy())
    expected_vflip_tensor.allclose(vfilp_tensor)

    diag_filp_tensor = flip_tensor(src_tensor, 'diagonal')
    expected_diag_filp_tensor = torch.from_numpy(img[..., ::-1, ::-1].copy())
    expected_diag_filp_tensor.allclose(diag_filp_tensor)


def test_select_single_mlvl():
    mlvl_tensors = [torch.rand(2, 1, 10, 10)] * 5
    mlvl_tensor_list = select_single_mlvl(mlvl_tensors, 1)
    assert len(mlvl_tensor_list) == 5 and mlvl_tensor_list[0].ndim == 3


def test_filter_scores_and_topk():
    score = torch.tensor([[0.1, 0.3, 0.2], [0.12, 0.7, 0.9], [0.02, 0.8, 0.08],
                          [0.4, 0.1, 0.08]])
    bbox_pred = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.1, 0.1], [0.5, 0.1]])
    score_thr = 0.15
    nms_pre = 4
    # test results type error
    with pytest.raises(NotImplementedError):
        filter_scores_and_topk(score, score_thr, nms_pre, (score, ))

    filtered_results = filter_scores_and_topk(
        score, score_thr, nms_pre, results=dict(bbox_pred=bbox_pred))
    filtered_score, labels, keep_idxs, results = filtered_results
    assert filtered_score.allclose(torch.tensor([0.9, 0.8, 0.7, 0.4]))
    assert labels.allclose(torch.tensor([2, 1, 1, 0]))
    assert keep_idxs.allclose(torch.tensor([1, 2, 1, 3]))
    assert results['bbox_pred'].allclose(
        torch.tensor([[0.4, 0.7], [0.1, 0.1], [0.4, 0.7], [0.5, 0.1]]))
