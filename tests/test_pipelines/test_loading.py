import copy
import os.path as osp

import numpy as np
import pytest

from mmdet.datasets.pipelines import (LoadImageFromFile,
                                      LoadMultiChannelImageFromFiles,
                                      LoadMultiImagesFromMultiFiles)


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

    def test_load_multi_imgs(self):
        results = dict(
            img_prefix=self.data_prefix,
            target_imgs_info=dict(filename='color.jpg'),
            ref_imgs_info=dict(filename=['color.jpg', 'color.jpg']))
        transform = LoadMultiImagesFromMultiFiles()
        results = transform(copy.deepcopy(results))
        assert results['target_filenames'] == [
            osp.join(self.data_prefix, 'color.jpg')
        ]
        assert results['ref_filenames'] == [
            osp.join(self.data_prefix, 'color.jpg'),
            osp.join(self.data_prefix, 'color.jpg')
        ]
        assert results['target_ori_filenames'] == ['color.jpg']
        assert results['ref_ori_filenames'] == ['color.jpg', 'color.jpg']
        assert results['target_img_0'].shape == (288, 512, 3)
        assert results['target_img_0'].dtype == np.uint8
        assert results['ref_img_0'].shape == (288, 512, 3)
        assert results['ref_img_0'].dtype == np.uint8
        assert results['ref_img_1'].shape == (288, 512, 3)
        assert results['ref_img_1'].dtype == np.uint8
        assert results['img_shape'] == (288, 512, 3)
        assert results['ori_shape'] == (288, 512, 3)
        assert results['pad_shape'] == (288, 512, 3)
        assert results['scale_factor'] == 1.0
        assert repr(transform) == transform.__class__.__name__ + \
            "(\n\tprefixs=['target', 'ref'],\n" + \
            '\tto_float32=False,\n' + \
            "\tcolor_type='color',\n" + \
            "\tfile_client_args={'backend': 'disk'})\n"

        # results not contain any element in img_info_keys
        results = dict(
            img_prefix=self.data_prefix,
            support_img_info=dict(filename='color.jpg'),
            reference_img_info=dict(filename=['color.jpg', 'color.jpg']))
        transform = LoadMultiImagesFromMultiFiles()
        # test assertion if results not contain any element in img_info_keys
        with pytest.raises(KeyError):
            transform(copy.deepcopy(results))

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
