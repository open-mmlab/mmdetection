<<<<<<< HEAD
# Copyright (c) OpenMMLab. All rights reserved.
=======
>>>>>>> [WIP] Add CrowdHumanDataset
import unittest

from mmdet.datasets import CrowdHumanDataset


class TestCrowdHumanDataset(unittest.TestCase):

    def test_crowdhuman_init(self):
        dataset = CrowdHumanDataset(
<<<<<<< HEAD
            data_root='tests/data/crowdhuman_dataset/',
            ann_file='test_annotation_train.odgt',
            data_prefix=dict(img='Images/'),
            pipeline=[])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.metainfo['CLASSES'], ('person', ))
=======
            data_root='tests/data/CrowdHuman/',
            ann_file='test_annotation_train.odgt',
            data_prefix=dict(img='Images/'),
            pipeline=[])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.metainfo['CLASSES'], ['person'])
>>>>>>> [WIP] Add CrowdHumanDataset
