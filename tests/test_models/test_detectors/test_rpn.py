# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet import *  # noqa
from mmdet.core import DetDataSample
from mmdet.testing import demo_mm_inputs, get_detector_cfg


class TestRPN(TestCase):

    @parameterized.expand(['rpn/rpn_r50_fpn_1x_coco.py'])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        detector = build_detector(model)
        assert detector.backbone
        assert detector.neck
        assert detector.rpn_head
        assert detector.device.type == 'cpu'

        # if rpn.num_classes > 1, force set rpn.num_classes = 1
        model.rpn_head.num_classes = 2
        detector = build_detector(model)
        assert detector.rpn_head.num_classes == 1

    @parameterized.expand([('rpn/rpn_r50_fpn_1x_coco.py', ('cpu', 'cuda'))])
    def test_rpn_forward_train(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = build_detector(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            assert detector.device.type == device

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])

            # Test forward train
            losses = detector.forward(packed_inputs, return_loss=True)
            assert isinstance(losses, dict)

            # Test forward_dummy
            batch = torch.ones((1, 3, 64, 64)).to(device=device)
            out = detector.forward_dummy(batch)
            assert isinstance(out, tuple)
            assert len(out) == 2

    @parameterized.expand([('rpn/rpn_r50_fpn_1x_coco.py', ('cpu', 'cuda'))])
    def test_single_stage_forward_test(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = build_detector(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            assert detector.device.type == device

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])

            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(
                    packed_inputs, return_loss=False)
                assert len(batch_results) == 2
                isinstance(batch_results[0], DetDataSample)
