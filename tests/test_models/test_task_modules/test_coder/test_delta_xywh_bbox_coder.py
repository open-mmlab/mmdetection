# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet.models.task_modules.coders import DeltaXYWHBBoxCoder


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
