from unittest import TestCase

import numpy as np
from mmengine.registry import init_default_scope

from mmdet.registry import TASK_UTILS


class TestKalmanFilter(TestCase):

    @classmethod
    def setUpClass(cls):
        init_default_scope('mmdet')
        motion = dict(type='KalmanFilter', )
        cls.kf = TASK_UTILS.build(motion)

    def test_init(self):
        pred_det = np.random.randn(4)
        mean, covariance = self.kf.initiate(pred_det)
        assert len(mean) == 8
        assert covariance.shape == (8, 8)

    def test_predict(self):
        mean = np.random.randn(8)
        covariance = np.random.randn(8, 8)
        mean, covariance = self.kf.predict(mean, covariance)
        assert len(mean) == 8
        assert covariance.shape == (8, 8)

    def test_update(self):
        mean = np.ones(8)
        covariance = np.ones((8, 8))
        measurement = np.ones(4)
        score = 0.1
        mean, covariance = self.kf.update(mean, covariance, measurement, score)
        assert len(mean) == 8
        assert covariance.shape == (8, 8)
