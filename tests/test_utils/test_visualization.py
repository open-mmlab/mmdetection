# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest
import torch

from mmdet.core import visualization as vis
from mmdet.datasets import (CityscapesDataset, CocoDataset,
                            CocoPanopticDataset, VOCDataset)


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
    out_image = vis.imshow_det_bboxes(
        image, bbox, label, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    assert image.shape == out_image.shape
    assert not np.allclose(image, out_image)
    os.remove(tmp_filename)

    # test grayscale images
    image = np.ones((10, 10), np.uint8)
    bbox = np.array([[2, 1, 3, 3], [3, 4, 6, 6]])
    label = np.array([0, 1])
    out_image = vis.imshow_det_bboxes(
        image, bbox, label, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    assert image.shape == out_image.shape[:2]
    os.remove(tmp_filename)

    # test shaped (0,)
    image = np.ones((10, 10, 3), np.uint8)
    bbox = np.ones((0, 4))
    label = np.ones((0, ))
    vis.imshow_det_bboxes(
        image, bbox, label, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    os.remove(tmp_filename)

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
    out_image = vis.imshow_gt_det_bboxes(
        image, annotation, result, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    assert image.shape == out_image.shape
    assert not np.allclose(image, out_image)
    os.remove(tmp_filename)

    # test grayscale images
    image = np.ones((10, 10), np.uint8)
    bbox = np.array([[2, 1, 3, 3], [3, 4, 6, 6]])
    label = np.array([0, 1])
    annotation = dict(gt_bboxes=bbox, gt_labels=label)
    det_result = np.array([[2, 1, 3, 3, 0], [3, 4, 6, 6, 1]])
    result = [det_result]
    vis.imshow_gt_det_bboxes(
        image, annotation, result, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    os.remove(tmp_filename)

    # test numpy mask
    gt_mask = np.ones((2, 10, 10))
    annotation['gt_masks'] = gt_mask
    vis.imshow_gt_det_bboxes(
        image, annotation, result, out_file=tmp_filename, show=False)
    assert osp.isfile(tmp_filename)
    os.remove(tmp_filename)

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


def test_palette():
    assert vis.palette_val([(1, 2, 3)])[0] == (1 / 255, 2 / 255, 3 / 255)

    # test list
    palette = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    palette_ = vis.get_palette(palette, 3)
    for color, color_ in zip(palette, palette_):
        assert color == color_

    # test tuple
    palette = vis.get_palette((1, 2, 3), 3)
    assert len(palette) == 3
    for color in palette:
        assert color == (1, 2, 3)

    # test color str
    palette = vis.get_palette('red', 3)
    assert len(palette) == 3
    for color in palette:
        assert color == (255, 0, 0)

    # test dataset str
    palette = vis.get_palette('coco', len(CocoDataset.CLASSES))
    assert len(palette) == len(CocoDataset.CLASSES)
    assert palette[0] == (220, 20, 60)
    palette = vis.get_palette('coco', len(CocoPanopticDataset.CLASSES))
    assert len(palette) == len(CocoPanopticDataset.CLASSES)
    assert palette[-1] == (250, 141, 255)
    palette = vis.get_palette('voc', len(VOCDataset.CLASSES))
    assert len(palette) == len(VOCDataset.CLASSES)
    assert palette[0] == (106, 0, 228)
    palette = vis.get_palette('citys', len(CityscapesDataset.CLASSES))
    assert len(palette) == len(CityscapesDataset.CLASSES)
    assert palette[0] == (220, 20, 60)

    # test random
    palette1 = vis.get_palette('random', 3)
    palette2 = vis.get_palette(None, 3)
    for color1, color2 in zip(palette1, palette2):
        assert isinstance(color1, tuple)
        assert isinstance(color2, tuple)
        assert color1 == color2
