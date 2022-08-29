# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmdet.datasets import WIDERFaceDataset


class TestWIDERFaceDataset(unittest.TestCase):

    def test_wider_face_dataset(self):
        dataset = WIDERFaceDataset(
            data_root='tests/data/WIDERFace',
            ann_file='train.txt',
            data_prefix=dict(img='WIDER_train'),
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)

        data_list = dataset.load_data_list()
        self.assertEqual(len(data_list), 1)
        self.assertEqual(len(data_list[0]['instances']), 10)
