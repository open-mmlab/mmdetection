# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
from mmengine.registry import init_default_scope
from torch import nn

from mmdet.registry import TASK_UTILS


class TestAppearanceFreeLink(TestCase):

    @classmethod
    def setUpClass(cls):
        init_default_scope('mmdet')
        cls.cfg = dict(
            type='AppearanceFreeLink',
            checkpoint='',
            temporal_threshold=(0, 30),
            spatial_threshold=75,
            confidence_threshold=0.95,
        )

    def test_init(self):
        aflink = TASK_UTILS.build(self.cfg)
        assert aflink.temporal_threshold == (0, 30)
        assert aflink.spatial_threshold == 75
        assert aflink.confidence_threshold == 0.95
        assert isinstance(aflink.model, nn.Module)

    def test_forward(self):
        pred_track = np.random.randn(10, 7)
        aflink = TASK_UTILS.build(self.cfg)
        linked_track = aflink.forward(pred_track)
        assert isinstance(linked_track, np.ndarray)
        assert linked_track.shape == (10, 7)
