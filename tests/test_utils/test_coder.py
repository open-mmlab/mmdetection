import pytest
import torch

from mmdet.core.bbox.coder import (DeltaXYWHBBoxCoder, TBLRBBoxCoder,
                                   YOLOBBoxCoder)


def test_yolo_bbox_coder():
    coder = YOLOBBoxCoder()
    bboxes = torch.Tensor([[-42., -29., 74., 61.], [-10., -29., 106., 61.],
                           [22., -29., 138., 61.], [54., -29., 170., 61.]])
    pred_bboxes = torch.Tensor([[0.4709, 0.6152, 0.1690, -0.4056],
                                [0.5399, 0.6653, 0.1162, -0.4162],
                                [0.4654, 0.6618, 0.1548, -0.4301],
                                [0.4786, 0.6197, 0.1896, -0.4479]])
    grid_size = 32
    expected_decode_bboxes = torch.Tensor(
        [[-53.6102, -10.3096, 83.7478, 49.6824],
         [-15.8700, -8.3901, 114.4236, 50.9693],
         [11.1822, -8.0924, 146.6034, 50.4476],
         [41.2068, -8.9232, 181.4236, 48.5840]])
    assert expected_decode_bboxes.allclose(
        coder.decode(bboxes, pred_bboxes, grid_size))


def test_delta_bbox_coder():
    coder = DeltaXYWHBBoxCoder()

    rois = torch.Tensor([[0., 0., 1., 1.], [0., 0., 1., 1.], [0., 0., 1., 1.],
                         [5., 5., 5., 5.]])
    deltas = torch.Tensor([[0., 0., 0., 0.], [1., 1., 1., 1.],
                           [0., 0., 2., -1.], [0.7, -1.9, -0.5, 0.3]])
    expected_decode_bboxes = torch.Tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                                           [0.1409, 0.1409, 2.8591, 2.8591],
                                           [0.0000, 0.3161, 4.1945, 0.6839],
                                           [5.0000, 5.0000, 5.0000, 5.0000]])

    out = coder.decode(rois, deltas, max_shape=(32, 32))
    assert expected_decode_bboxes.allclose(out, atol=1e-04)
    out = coder.decode(rois, deltas, max_shape=torch.Tensor((32, 32)))
    assert expected_decode_bboxes.allclose(out, atol=1e-04)

    batch_rois = rois.unsqueeze(0).repeat(2, 1, 1)
    batch_deltas = deltas.unsqueeze(0).repeat(2, 1, 1)
    batch_out = coder.decode(batch_rois, batch_deltas, max_shape=(32, 32))[0]
    assert out.allclose(batch_out)
    batch_out = coder.decode(
        batch_rois, batch_deltas, max_shape=[(32, 32), (32, 32)])[0]
    assert out.allclose(batch_out)

    # test max_shape is not equal to batch
    with pytest.raises(AssertionError):
        coder.decode(
            batch_rois, batch_deltas, max_shape=[(32, 32), (32, 32), (32, 32)])

    rois = torch.zeros((0, 4))
    deltas = torch.zeros((0, 4))
    out = coder.decode(rois, deltas, max_shape=(32, 32))
    assert rois.shape == out.shape

    # test add_ctr_clamp
    coder = DeltaXYWHBBoxCoder(add_ctr_clamp=True, ctr_clamp=2)

    rois = torch.Tensor([[0., 0., 6., 6.], [0., 0., 1., 1.], [0., 0., 1., 1.],
                         [5., 5., 5., 5.]])
    deltas = torch.Tensor([[1., 1., 2., 2.], [1., 1., 1., 1.],
                           [0., 0., 2., -1.], [0.7, -1.9, -0.5, 0.3]])
    expected_decode_bboxes = torch.Tensor([[0.0000, 0.0000, 27.1672, 27.1672],
                                           [0.1409, 0.1409, 2.8591, 2.8591],
                                           [0.0000, 0.3161, 4.1945, 0.6839],
                                           [5.0000, 5.0000, 5.0000, 5.0000]])

    out = coder.decode(rois, deltas, max_shape=(32, 32))
    assert expected_decode_bboxes.allclose(out, atol=1e-04)


def test_tblr_bbox_coder():
    coder = TBLRBBoxCoder(normalizer=15.)

    rois = torch.Tensor([[0., 0., 1., 1.], [0., 0., 1., 1.], [0., 0., 1., 1.],
                         [5., 5., 5., 5.]])
    deltas = torch.Tensor([[0., 0., 0., 0.], [1., 1., 1., 1.],
                           [0., 0., 2., -1.], [0.7, -1.9, -0.5, 0.3]])
    expected_decode_bboxes = torch.Tensor([[0.5000, 0.5000, 0.5000, 0.5000],
                                           [0.0000, 0.0000, 12.0000, 13.0000],
                                           [0.0000, 0.5000, 0.0000, 0.5000],
                                           [5.0000, 5.0000, 5.0000, 5.0000]])

    out = coder.decode(rois, deltas, max_shape=(13, 12))
    assert expected_decode_bboxes.allclose(out)
    out = coder.decode(rois, deltas, max_shape=torch.Tensor((13, 12)))
    assert expected_decode_bboxes.allclose(out)

    batch_rois = rois.unsqueeze(0).repeat(2, 1, 1)
    batch_deltas = deltas.unsqueeze(0).repeat(2, 1, 1)
    batch_out = coder.decode(batch_rois, batch_deltas, max_shape=(13, 12))[0]
    assert out.allclose(batch_out)
    batch_out = coder.decode(
        batch_rois, batch_deltas, max_shape=[(13, 12), (13, 12)])[0]
    assert out.allclose(batch_out)

    # test max_shape is not equal to batch
    with pytest.raises(AssertionError):
        coder.decode(batch_rois, batch_deltas, max_shape=[(13, 12)])

    rois = torch.zeros((0, 4))
    deltas = torch.zeros((0, 4))
    out = coder.decode(rois, deltas, max_shape=(32, 32))
    assert rois.shape == out.shape
