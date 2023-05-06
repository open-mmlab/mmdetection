# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet.registry import MODELS
from mmdet.structures import ReIDDataSample
from mmdet.testing import get_detector_cfg
from mmdet.utils import register_all_modules


class TestBaseReID(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        register_all_modules()

    @parameterized.expand([
        'reid/reid_r50_8xb32-6e_mot17train80_test-mot17val20.py',
    ])
    def test_forward(self, cfg_file):
        model_cfg = get_detector_cfg(cfg_file)
        model = MODELS.build(model_cfg)
        inputs = torch.rand(1, 4, 3, 256, 128)
        data_samples = [
            ReIDDataSample().set_gt_label(label) for label in (0, 0, 1, 1)
        ]

        # test mode='tensor'
        feats = model(inputs, mode='tensor')
        assert feats.shape == (4, 128)

        # test mode='loss'
        losses = model(inputs, data_samples, mode='loss')
        assert losses.keys() == {'triplet_loss', 'ce_loss', 'accuracy_top-1'}
        assert losses['ce_loss'].item() > 0
        assert losses['triplet_loss'].item() > 0

        # test mode='predict'
        predictions = model(inputs, data_samples, mode='predict')
        for pred in predictions:
            assert isinstance(pred, ReIDDataSample)
            assert isinstance(pred.pred_feature, torch.Tensor)
            assert isinstance(pred.gt_label.label, torch.Tensor)
            assert pred.pred_feature.shape == (128, )
