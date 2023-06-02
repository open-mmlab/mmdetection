# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmdet.datasets import CrowdHumanDataset


class TestCrowdHumanDataset(unittest.TestCase):

    def test_crowdhuman_init(self):
        dataset = CrowdHumanDataset(
            data_root='tests/data/crowdhuman_dataset/',
            ann_file='test_annotation_train.odgt',
            data_prefix=dict(img='Images/'),
            pipeline=[])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.metainfo['classes'], ('person', ))
