# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
from mmengine.registry import init_default_scope

from mmdet.registry import TASK_UTILS


class TestInterpolateTracklets(TestCase):

    @classmethod
    def setUpClass(cls):
        init_default_scope('mmdet')
        cls.cfg = dict(
            type='InterpolateTracklets',
            min_num_frames=5,
            max_num_frames=20,
            use_gsi=True,
            smooth_tau=10)

    def test_init(self):
        interpolation = TASK_UTILS.build(self.cfg)
        assert interpolation.min_num_frames == 5
        assert interpolation.max_num_frames == 20
        assert interpolation.use_gsi
        assert interpolation.smooth_tau == 10

    def test_forward(self):
        pred_track = np.random.randn(5, 7)

        # set frame_id and target_id
        pred_track[:, 0] = np.array([1, 2, 5, 6, 7])
        pred_track[:, 1] = 1

        interpolation = TASK_UTILS.build(self.cfg)
        linked_track = interpolation.forward(pred_track)
        assert isinstance(linked_track, np.ndarray)
        assert linked_track.shape == (5, 7)
