# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import cv2
import numpy as np

from mmdet.datasets import WIDERFaceDataset


class TestWIDERFaceDataset(unittest.TestCase):

    def setUp(self) -> None:
        img_path = 'tests/data/WIDERFace/WIDER_train/0--Parade/0_Parade_marchingband_1_5.jpg'  # noqa: E501
        dummy_img = np.zeros((683, 1024, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy_img)

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
