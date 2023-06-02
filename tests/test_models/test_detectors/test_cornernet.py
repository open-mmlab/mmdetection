# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.config import ConfigDict

from mmdet.structures import DetDataSample
from mmdet.testing import demo_mm_inputs, get_detector_cfg
from mmdet.utils import register_all_modules


class TestCornerNet(TestCase):

    def setUp(self) -> None:
        register_all_modules()
        model_cfg = get_detector_cfg(
            'cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py')

        backbone = dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch')

        neck = dict(
            type='FPN',
            in_channels=[512],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=1)

        model_cfg.backbone = ConfigDict(**backbone)
        model_cfg.neck = ConfigDict(**neck)
        model_cfg.bbox_head.num_feat_levels = 1
        self.model_cfg = model_cfg

    def test_init(self):
        model = get_detector_cfg(
            'cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py')
        model.backbone.init_cfg = None

        from mmdet.registry import MODELS
        detector = MODELS.build(model)
        self.assertTrue(detector.bbox_head is not None)
        self.assertTrue(detector.backbone is not None)
        self.assertTrue(not hasattr(detector, 'neck'))

    @unittest.skipIf(not torch.cuda.is_available(),
                     'test requires GPU and torch+cuda')
    def test_cornernet_forward_loss_mode(self):
        from mmdet.registry import MODELS
        detector = MODELS.build(self.model_cfg)
        detector.init_weights()

        packed_inputs = demo_mm_inputs(2, [[3, 511, 511], [3, 511, 511]])
        data = detector.data_preprocessor(packed_inputs, True)
        losses = detector.forward(**data, mode='loss')
        assert isinstance(losses, dict)

    @unittest.skipIf(not torch.cuda.is_available(),
                     'test requires GPU and torch+cuda')
    def test_cornernet_forward_predict_mode(self):
        from mmdet.registry import MODELS
        detector = MODELS.build(self.model_cfg)
        detector.init_weights()

        packed_inputs = demo_mm_inputs(2, [[3, 512, 512], [3, 512, 512]])
        data = detector.data_preprocessor(packed_inputs, False)

        # Test forward test
        detector.eval()
        with torch.no_grad():
            batch_results = detector.forward(**data, mode='predict')
            assert len(batch_results) == 2
            assert isinstance(batch_results[0], DetDataSample)

    @unittest.skipIf(not torch.cuda.is_available(),
                     'test requires GPU and torch+cuda')
    def test_cornernet_forward_tensor_mode(self):
        from mmdet.registry import MODELS
        detector = MODELS.build(self.model_cfg)
        detector.init_weights()

        packed_inputs = demo_mm_inputs(2, [[3, 512, 512], [3, 512, 512]])
        data = detector.data_preprocessor(packed_inputs, False)
        batch_results = detector.forward(**data, mode='tensor')
        assert isinstance(batch_results, tuple)
