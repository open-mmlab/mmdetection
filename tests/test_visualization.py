# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest
import torch

from mmdet.core import visualization as vis


def test_color():
    assert vis.color_val_matplotlib(mmcv.Color.blue) == (0., 0., 1.)
    assert vis.color_val_matplotlib('green') == (0., 1., 0.)
    assert vis.color_val_matplotlib((1, 2, 3)) == (3 / 255, 2 / 255, 1 / 255)
    assert vis.color_val_matplotlib(100) == (100 / 255, 100 / 255, 100 / 255)
    assert vis.color_val_matplotlib(np.zeros(3, dtype=np.int)) == (0., 0., 0.)
    # forbid white color
    with pytest.raises(TypeError):
        vis.color_val_matplotlib([255, 255, 255])
    # forbid float
    with pytest.raises(TypeError):
        vis.color_val_matplotlib(1.0)
    # overflowed
    with pytest.raises(AssertionError):
        vis.color_val_matplotlib((0, 0, 500))


def test_imshow_det_bboxes():
    tmp_filename = osp.join(tempfile.gettempdir(), 'det_bboxes_image',
                            'image.jpg')
    image = np.ones((10, 10, 3), np.uint8)
    bbox = np.array([[2, 1, 3, 3], [3, 4, 6, 6]])
    label = np.array([0, 1])
    vis.imshow_det_bboxes(
        image, bbox, label, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)

    # test shaped (0,)
    image = np.ones((10, 10, 3), np.uint8)
    bbox = np.ones((0, 4))
    label = np.ones((0, ))
    vis.imshow_det_bboxes(
        image, bbox, label, out_file=tmp_filename, show=False)

    # test mask
    image = np.ones((10, 10, 3), np.uint8)
    bbox = np.array([[2, 1, 3, 3], [3, 4, 6, 6]])
    label = np.array([0, 1])
    segms = np.random.random((2, 10, 10)) > 0.5
    segms = np.array(segms, np.int32)
    vis.imshow_det_bboxes(
        image, bbox, label, segms, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)

    os.remove(tmp_filename)

    # test tensor mask type error
    with pytest.raises(AttributeError):
        segms = torch.tensor(segms)
        vis.imshow_det_bboxes(image, bbox, label, segms, show=False)


def test_imshow_gt_det_bboxes():
    tmp_filename = osp.join(tempfile.gettempdir(), 'det_bboxes_image',
                            'image.jpg')
    image = np.ones((10, 10, 3), np.uint8)
    bbox = np.array([[2, 1, 3, 3], [3, 4, 6, 6]])
    label = np.array([0, 1])
    annotation = dict(gt_bboxes=bbox, gt_labels=label)
    det_result = np.array([[2, 1, 3, 3, 0], [3, 4, 6, 6, 1]])
    result = [det_result]
    vis.imshow_gt_det_bboxes(
        image, annotation, result, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)

    # test numpy mask
    gt_mask = np.ones((2, 10, 10))
    annotation['gt_masks'] = gt_mask
    vis.imshow_gt_det_bboxes(
        image, annotation, result, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)

    # test tensor mask
    gt_mask = torch.ones((2, 10, 10))
    annotation['gt_masks'] = gt_mask
    vis.imshow_gt_det_bboxes(
        image, annotation, result, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)

    os.remove(tmp_filename)

    # test unsupported type
    annotation['gt_masks'] = []
    with pytest.raises(TypeError):
        vis.imshow_gt_det_bboxes(image, annotation, result, show=False)
