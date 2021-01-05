# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest

from mmdet.core import visualization as vis


def test_color():
    assert vis.color_val_matplotlib(mmcv.Color.blue) == (0., 0., 1.)
    assert vis.color_val_matplotlib('green') == (0., 1., 0.)
    assert vis.color_val_matplotlib((1, 2, 3)) == (3 / 255, 2 / 255, 1 / 255)
    assert vis.color_val_matplotlib(100) == (100 / 255, 100 / 255, 100 / 255)
    assert vis.color_val_matplotlib(np.zeros(3, dtype=np.int)) == (0., 0., 0.)
    with pytest.raises(TypeError):
        vis.color_val_matplotlib([255, 255, 255])
    with pytest.raises(TypeError):
        vis.color_val_matplotlib(1.0)
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
