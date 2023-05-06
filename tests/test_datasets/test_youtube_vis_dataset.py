# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmdet.datasets import YouTubeVISDataset


class TestYouTubeVISDataset(TestCase):

    @classmethod
    def setUpClass(cls):

        cls.dataset = YouTubeVISDataset(
            ann_file='tests/data/vis_sample.json', dataset_version='2019')

    def test_set_dataset_classes(self):
        assert isinstance(self.dataset.metainfo, dict)
        assert len(self.dataset.metainfo['classes']) == 40
