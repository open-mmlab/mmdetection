# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmdet.datasets import OpenImagesChallengeDataset, OpenImagesDataset


class TestOpenImagesDataset(unittest.TestCase):

    def test_init(self):
        dataset = OpenImagesDataset(
            data_root='tests/data/OpenImages/',
            ann_file='annotations/oidv6-train-annotations-bbox.csv',
            data_prefix=dict(img='OpenImages/train/'),
            label_file='annotations/class-descriptions-boxable.csv',
            hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
            meta_file='annotations/image-metas.pkl',
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.metainfo['classes'], ['Airplane'])


class TestOpenImagesChallengeDataset(unittest.TestCase):

    def test_init(self):
        dataset = OpenImagesChallengeDataset(
            data_root='tests/data/OpenImages/',
            ann_file='challenge2019/challenge-2019-train-detection-bbox.txt',
            data_prefix=dict(img='OpenImages/train/'),
            label_file='challenge2019/cls-label-description.csv',
            hierarchy_file='challenge2019/class_label_tree.np',
            meta_file='annotations/image-metas.pkl',
            pipeline=[])
        dataset.full_init()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.metainfo['classes'], ['Airplane'])
