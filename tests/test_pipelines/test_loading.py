import copy
import os.path as osp

import numpy as np

from mmdet.datasets.pipelines import (LoadImageFromFile,
                                      LoadMultiChannelImageFromFiles)


class TestLoading(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../data')

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
        assert results['pad_shape'] == (288, 512, 3)
        assert results['scale_factor'] == 1.0
        np.testing.assert_equal(results['img_norm_cfg']['mean'],
                                np.zeros(3, dtype=np.float32))
        assert repr(transform) == transform.__class__.__name__ + \
            "(to_float32=False, color_type='color', " + \
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
        np.testing.assert_equal(results['img_norm_cfg']['mean'],
                                np.zeros(1, dtype=np.float32))

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
