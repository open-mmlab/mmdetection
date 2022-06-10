# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet import *  # noqa
from mmdet.core import DetDataSample
from mmdet.testing import demo_mm_inputs, get_detector_cfg


class TestTwoStageBBox(TestCase):

    @parameterized.expand([
        'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        'cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    ])
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
        assert detector.roi_head
        assert detector.device.type == 'cpu'

        # if rpn.num_classes > 1, force set rpn.num_classes = 1
        model.rpn_head.num_classes = 2
        detector = build_detector(model)
        assert detector.rpn_head.num_classes == 1

    @parameterized.expand([
        ('faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', 'cuda'),
        ('cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py', 'cuda')
    ])
    def test_two_stage_forward_train(self, cfg_file, device):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        detector = build_detector(model)

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

    @parameterized.expand([
        ('faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', 'cuda'),
        ('cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py', 'cuda')
    ])
    def test_two_stage_forward_test(self, cfg_file, device):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        detector = build_detector(model)

        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.cuda()

        assert detector.device.type == device

        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])

        # Test forward test
        detector.eval()
        with torch.no_grad():
            batch_results = detector.forward(packed_inputs, return_loss=False)
            assert len(batch_results) == 2
            assert isinstance(batch_results[0], DetDataSample)


class TestTwoStageMask(TestCase):

    @parameterized.expand([
        'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
        'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
    ])
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
        assert detector.roi_head
        assert detector.roi_head.mask_head
        assert detector.device.type == 'cpu'

    @parameterized.expand([
        ('mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py', 'cuda'),
        ('cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py', 'cuda')
    ])
    def test_forward_train(self, cfg_file, device):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        detector = build_detector(model)

        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.cuda()

        assert detector.device.type == device

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 125, 130]], with_mask=True)

        # Test forward train
        losses = detector.forward(packed_inputs, return_loss=True)
        assert isinstance(losses, dict)

        # Test forward_dummy
        batch = torch.ones((1, 3, 64, 64)).to(device=device)
        out = detector.forward_dummy(batch)
        assert isinstance(out, tuple)

    @parameterized.expand([
        ('mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py', 'cuda'),
        ('cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py', 'cuda')
    ])
    def test_forward_test(self, cfg_file, device):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.backbone.depth = 18
        model.neck.in_channels = [64, 128, 256, 512]
        model.backbone.init_cfg = None

        from mmdet.models import build_detector
        detector = build_detector(model)

        if not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.cuda()

        assert detector.device.type == device

        packed_inputs = demo_mm_inputs(2, [[3, 256, 256], [3, 255, 260]])

        # Test forward test
        detector.eval()
        with torch.no_grad():
            batch_results = detector.forward(packed_inputs, return_loss=False)
            assert len(batch_results) == 2
            assert isinstance(batch_results[0], DetDataSample)
