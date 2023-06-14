# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmdet.datasets import MOTChallengeDataset


class TestMOTChallengeDataset(unittest.TestCase):

    def test_mot_challenge_dataset(self):
        # test CocoDataset
        metainfo = dict(classes=('pedestrian'), task_name='new_task')
        dataset = MOTChallengeDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/mot_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        self.assertEqual(dataset.metainfo['classes'], ('pedestrian'))
        self.assertEqual(dataset.metainfo['task_name'], 'new_task')
        self.assertListEqual(dataset.get_cat_ids((0, 1)), [0, 0])
        self.assertListEqual(dataset.get_cat_ids(0), [0, 0, 0, 0, 0, 0])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.num_all_imgs, 5)
        self.assertEqual(len(dataset[0]['images'][2]['instances']), 2)

    def test_mot_challenge_dataset_with_visibility(self):
        dataset = MOTChallengeDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/mot_sample.json',
            metainfo=dict(classes=('pedestrian')),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            visibility_thr=0.5,
            pipeline=[])
        self.assertEqual(dataset.num_all_imgs, 5)
        self.assertEqual(len(dataset[0]['images'][2]['instances']), 1)
