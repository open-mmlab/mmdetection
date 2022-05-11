# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdet.core import BboxOverlaps2D, bbox_overlaps
from mmdet.core.evaluation.bbox_overlaps import \
    bbox_overlaps as recall_overlaps


def test_bbox_overlaps_2d(eps=1e-7):

    def _construct_bbox(num_bbox=None):
        img_h = int(np.random.randint(3, 1000))
        img_w = int(np.random.randint(3, 1000))
        if num_bbox is None:
            num_bbox = np.random.randint(1, 10)
        x1y1 = torch.rand((num_bbox, 2))
        x2y2 = torch.max(torch.rand((num_bbox, 2)), x1y1)
        bboxes = torch.cat((x1y1, x2y2), -1)
        bboxes[:, 0::2] *= img_w
        bboxes[:, 1::2] *= img_h
        return bboxes, num_bbox

    # is_aligned is True, bboxes.size(-1) == 5 (include score)
    self = BboxOverlaps2D()
    bboxes1, num_bbox = _construct_bbox()
    bboxes2, _ = _construct_bbox(num_bbox)
    bboxes1 = torch.cat((bboxes1, torch.rand((num_bbox, 1))), 1)
    bboxes2 = torch.cat((bboxes2, torch.rand((num_bbox, 1))), 1)
    gious = self(bboxes1, bboxes2, 'giou', True)
    assert gious.size() == (num_bbox, ), gious.size()
    assert torch.all(gious >= -1) and torch.all(gious <= 1)

    # is_aligned is True, bboxes1.size(-2) == 0
    bboxes1 = torch.empty((0, 4))
    bboxes2 = torch.empty((0, 4))
    gious = self(bboxes1, bboxes2, 'giou', True)
    assert gious.size() == (0, ), gious.size()
    assert torch.all(gious == torch.empty((0, )))
    assert torch.all(gious >= -1) and torch.all(gious <= 1)

    # is_aligned is True, and bboxes.ndims > 2
    bboxes1, num_bbox = _construct_bbox()
    bboxes2, _ = _construct_bbox(num_bbox)
    bboxes1 = bboxes1.unsqueeze(0).repeat(2, 1, 1)
    # test assertion when batch dim is not the same
    with pytest.raises(AssertionError):
        self(bboxes1, bboxes2.unsqueeze(0).repeat(3, 1, 1), 'giou', True)
    bboxes2 = bboxes2.unsqueeze(0).repeat(2, 1, 1)
    gious = self(bboxes1, bboxes2, 'giou', True)
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (2, num_bbox)
    bboxes1 = bboxes1.unsqueeze(0).repeat(2, 1, 1, 1)
    bboxes2 = bboxes2.unsqueeze(0).repeat(2, 1, 1, 1)
    gious = self(bboxes1, bboxes2, 'giou', True)
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (2, 2, num_bbox)

    # is_aligned is False
    bboxes1, num_bbox1 = _construct_bbox()
    bboxes2, num_bbox2 = _construct_bbox()
    gious = self(bboxes1, bboxes2, 'giou')
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (num_bbox1, num_bbox2)

    # is_aligned is False, and bboxes.ndims > 2
    bboxes1 = bboxes1.unsqueeze(0).repeat(2, 1, 1)
    bboxes2 = bboxes2.unsqueeze(0).repeat(2, 1, 1)
    gious = self(bboxes1, bboxes2, 'giou')
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (2, num_bbox1, num_bbox2)
    bboxes1 = bboxes1.unsqueeze(0)
    bboxes2 = bboxes2.unsqueeze(0)
    gious = self(bboxes1, bboxes2, 'giou')
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (1, 2, num_bbox1, num_bbox2)

    # is_aligned is False, bboxes1.size(-2) == 0
    gious = self(torch.empty(1, 2, 0, 4), bboxes2, 'giou')
    assert torch.all(gious == torch.empty(1, 2, 0, bboxes2.size(-2)))
    assert torch.all(gious >= -1) and torch.all(gious <= 1)

    # test allclose between bbox_overlaps and the original official
    # implementation.
    bboxes1 = torch.FloatTensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [32, 32, 38, 42],
    ])
    bboxes2 = torch.FloatTensor([
        [0, 0, 10, 20],
        [0, 10, 10, 19],
        [10, 10, 20, 20],
    ])
    gious = bbox_overlaps(bboxes1, bboxes2, 'giou', is_aligned=True, eps=eps)
    gious = gious.numpy().round(4)
    # the gt is got with four decimal precision.
    expected_gious = np.array([0.5000, -0.0500, -0.8214])
    assert np.allclose(gious, expected_gious, rtol=0, atol=eps)

    # test mode 'iof'
    ious = bbox_overlaps(bboxes1, bboxes2, 'iof', is_aligned=True, eps=eps)
    assert torch.all(ious >= -1) and torch.all(ious <= 1)
    assert ious.size() == (bboxes1.size(0), )
    ious = bbox_overlaps(bboxes1, bboxes2, 'iof', eps=eps)
    assert torch.all(ious >= -1) and torch.all(ious <= 1)
    assert ious.size() == (bboxes1.size(0), bboxes2.size(0))


def test_voc_recall_overlaps():

    def _construct_bbox(num_bbox=None):
        img_h = int(np.random.randint(3, 1000))
        img_w = int(np.random.randint(3, 1000))
        if num_bbox is None:
            num_bbox = np.random.randint(1, 10)
        x1y1 = torch.rand((num_bbox, 2))
        x2y2 = torch.max(torch.rand((num_bbox, 2)), x1y1)
        bboxes = torch.cat((x1y1, x2y2), -1)
        bboxes[:, 0::2] *= img_w
        bboxes[:, 1::2] *= img_h
        return bboxes.numpy(), num_bbox

    bboxes1, num_bbox = _construct_bbox()
    bboxes2, _ = _construct_bbox(num_bbox)
    ious = recall_overlaps(
        bboxes1, bboxes2, 'iou', use_legacy_coordinate=False)
    assert ious.shape == (num_bbox, num_bbox)
    assert np.all(ious >= -1) and np.all(ious <= 1)

    ious = recall_overlaps(bboxes1, bboxes2, 'iou', use_legacy_coordinate=True)
    assert ious.shape == (num_bbox, num_bbox)
    assert np.all(ious >= -1) and np.all(ious <= 1)
