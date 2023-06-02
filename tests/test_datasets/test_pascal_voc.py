# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmdet.datasets import VOCDataset


class TestVOCDataset(unittest.TestCase):

    def test_voc2007_init(self):
        dataset = VOCDataset(
            data_root='tests/data/VOCdevkit/',
            ann_file='VOC2007/ImageSets/Main/trainval.txt',
            data_prefix=dict(sub_data_root='VOC2007/'),
            filter_cfg=dict(
                filter_empty_gt=True, min_size=32, bbox_min_size=32),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)

        data_list = dataset.load_data_list()
        self.assertEqual(len(data_list), 1)
        self.assertEqual(len(data_list[0]['instances']), 2)
        self.assertEqual(dataset.get_cat_ids(0), [11, 14])

    def test_voc2012_init(self):
        dataset = VOCDataset(
            data_root='tests/data/VOCdevkit/',
            ann_file='VOC2012/ImageSets/Main/trainval.txt',
            data_prefix=dict(sub_data_root='VOC2012/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)

        data_list = dataset.load_data_list()
        self.assertEqual(len(data_list), 1)
        self.assertEqual(len(data_list[0]['instances']), 1)
        self.assertEqual(dataset.get_cat_ids(0), [18])
