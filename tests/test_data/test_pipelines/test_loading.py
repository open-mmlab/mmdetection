# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np
import pytest

from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets.pipelines import (FilterAnnotations, LoadImageFromFile,
                                      LoadImageFromWebcam,
                                      LoadMultiChannelImageFromFiles)


class TestLoading:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../../data')

    def test_load_img(self):
        results = dict(
            img_prefix=self.data_prefix, img_info=dict(filename='color.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == osp.join(self.data_prefix, 'color.jpg')
        assert results['ori_filename'] == 'color.jpg'
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (288, 512, 3)
        assert results['ori_shape'] == (288, 512, 3)
        assert repr(transform) == transform.__class__.__name__ + \
            "(to_float32=False, color_type='color', channel_order='bgr', " + \
            "file_client_args={'backend': 'disk'})"

        # no img_prefix
        results = dict(
            img_prefix=None, img_info=dict(filename='tests/data/color.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == 'tests/data/color.jpg'
        assert results['ori_filename'] == 'tests/data/color.jpg'
        assert results['img'].shape == (288, 512, 3)

        # to_float32
        transform = LoadImageFromFile(to_float32=True)
        results = transform(copy.deepcopy(results))
        assert results['img'].dtype == np.float32

        # gray image
        results = dict(
            img_prefix=self.data_prefix, img_info=dict(filename='gray.jpg'))
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8

        transform = LoadImageFromFile(color_type='unchanged')
        results = transform(copy.deepcopy(results))
        assert results['img'].shape == (288, 512)
        assert results['img'].dtype == np.uint8

    def test_load_multi_channel_img(self):
        results = dict(
            img_prefix=self.data_prefix,
            img_info=dict(filename=['color.jpg', 'color.jpg']))
        transform = LoadMultiChannelImageFromFiles()
        results = transform(copy.deepcopy(results))
        assert results['filename'] == [
            osp.join(self.data_prefix, 'color.jpg'),
            osp.join(self.data_prefix, 'color.jpg')
        ]
        assert results['ori_filename'] == ['color.jpg', 'color.jpg']
        assert results['img'].shape == (288, 512, 3, 2)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (288, 512, 3, 2)
        assert results['ori_shape'] == (288, 512, 3, 2)
        assert results['pad_shape'] == (288, 512, 3, 2)
        assert results['scale_factor'] == 1.0
        assert repr(transform) == transform.__class__.__name__ + \
            "(to_float32=False, color_type='unchanged', " + \
            "file_client_args={'backend': 'disk'})"

    def test_load_webcam_img(self):
        img = mmcv.imread(osp.join(self.data_prefix, 'color.jpg'))
        results = dict(img=img)
        transform = LoadImageFromWebcam()
        results = transform(copy.deepcopy(results))
        assert results['filename'] is None
        assert results['ori_filename'] is None
        assert results['img'].shape == (288, 512, 3)
        assert results['img'].dtype == np.uint8
        assert results['img_shape'] == (288, 512, 3)
        assert results['ori_shape'] == (288, 512, 3)


def _build_filter_annotations_args():
    kwargs = (dict(min_gt_bbox_wh=(100, 100)),
              dict(min_gt_bbox_wh=(100, 100), keep_empty=False),
              dict(min_gt_bbox_wh=(1, 1)), dict(min_gt_bbox_wh=(.01, .01)),
              dict(min_gt_bbox_wh=(.01, .01),
                   by_mask=True), dict(by_mask=True),
              dict(by_box=False, by_mask=True))
    targets = (None, 0, 1, 2, 1, 1, 1)

    return list(zip(targets, kwargs))


@pytest.mark.parametrize('target, kwargs', _build_filter_annotations_args())
def test_filter_annotations(target, kwargs):
    filter_ann = FilterAnnotations(**kwargs)
    bboxes = np.array([[2., 10., 4., 14.], [2., 10., 2.1, 10.1]])
    raw_masks = np.zeros((2, 24, 24))
    raw_masks[0, 10:14, 2:4] = 1
    bitmap_masks = BitmapMasks(raw_masks, 24, 24)
    results = dict(gt_bboxes=bboxes, gt_masks=bitmap_masks)
    results = filter_ann(results)
    if results is not None:
        results = results['gt_bboxes'].shape[0]
    assert results == target

    polygons = [[np.array([2.0, 10.0, 4.0, 10.0, 4.0, 14.0, 2.0, 14.0])],
                [np.array([2.0, 10.0, 2.1, 10.0, 2.1, 10.1, 2.0, 10.1])]]
    polygon_masks = PolygonMasks(polygons, 24, 24)

    results = dict(gt_bboxes=bboxes, gt_masks=polygon_masks)
    results = filter_ann(results)

    if results is not None:
        results = len(results.get('gt_masks').masks)

    assert results == target
