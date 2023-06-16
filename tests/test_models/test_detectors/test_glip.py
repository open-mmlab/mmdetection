# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet.structures import DetDataSample
from mmdet.testing import demo_mm_inputs, get_detector_cfg
from mmdet.utils import register_all_modules


class TestGLIP(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand(
        ['glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py'])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmdet.registry import MODELS
        detector = MODELS.build(model)
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.language_model)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.bbox_head)

    @parameterized.expand([
        ('glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py', ('cpu',
                                                                   'cuda'))
    ])
    def test_glip_forward_predict_mode(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmdet.registry import MODELS
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = MODELS.build(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            # test custom_entities is True
            packed_inputs = demo_mm_inputs(
                2, [[3, 128, 128], [3, 125, 130]],
                texts=['a', 'b'],
                custom_entities=True)
            data = detector.data_preprocessor(packed_inputs, False)
            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(**data, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], DetDataSample)

            # test custom_entities is False
            packed_inputs = demo_mm_inputs(
                2, [[3, 128, 128], [3, 125, 130]],
                texts=['a', 'b'],
                custom_entities=False)
            data = detector.data_preprocessor(packed_inputs, False)
            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(**data, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], DetDataSample)
