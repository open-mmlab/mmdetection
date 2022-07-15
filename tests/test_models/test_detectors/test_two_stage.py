# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet.data_elements import DetDataSample
from mmdet.testing import demo_mm_inputs, get_detector_cfg
from mmdet.utils import register_all_modules


class TestTwoStageBBox(TestCase):

    def setUp(self):
        register_all_modules()

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
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.rpn_head)
        self.assertTrue(detector.roi_head)

        # if rpn.num_classes > 1, force set rpn.num_classes = 1
        model.rpn_head.num_classes = 2
        detector = build_detector(model)
        self.assertEqual(detector.rpn_head.num_classes, 1)

    @parameterized.expand([
        'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        'cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    ])
    def test_two_stage_forward_loss_mode(self, cfg_file):
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

        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])

        batch_inputs, data_samples = detector.data_preprocessor(
            packed_inputs, True)
        # Test loss mode
        losses = detector.forward(batch_inputs, data_samples, mode='loss')
        self.assertIsInstance(losses, dict)

    @parameterized.expand([
        'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        'cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    ])
    def test_single_stage_forward_predict_mode(self, cfg_file):
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

        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
        batch_inputs, data_samples = detector.data_preprocessor(
            packed_inputs, False)
        # Test forward test
        detector.eval()
        with torch.no_grad():
            with torch.no_grad():
                batch_results = detector.forward(
                    batch_inputs, data_samples, mode='predict')
            self.assertEqual(len(batch_results), 2)
            self.assertIsInstance(batch_results[0], DetDataSample)

    @parameterized.expand([
        'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        'cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
    ])
    def test_single_stage_forward_tensor_mode(self, cfg_file):
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

        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])

        batch_inputs, data_samples = detector.data_preprocessor(
            packed_inputs, False)

        # TODO: Awaiting refactoring
        # out = detector.forward(batch_inputs, data_samples, mode='tensor')
        # self.assertIsInstance(out, tuple)


class TestTwoStageMask(TestCase):

    def setUp(self):
        register_all_modules()

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
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.rpn_head)
        self.assertTrue(detector.roi_head)
        self.assertTrue(detector.roi_head.mask_head)

        # if rpn.num_classes > 1, force set rpn.num_classes = 1
        model.rpn_head.num_classes = 2
        detector = build_detector(model)
        self.assertEqual(detector.rpn_head.num_classes, 1)

    @parameterized.expand([
        'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
        'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
    ])
    def test_single_stage_forward_loss_mode(self, cfg_file):
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

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 125, 130]], with_mask=True)
        batch_inputs, data_samples = detector.data_preprocessor(
            packed_inputs, True)
        # Test loss mode
        losses = detector.forward(batch_inputs, data_samples, mode='loss')
        self.assertIsInstance(losses, dict)

    @parameterized.expand([
        'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
        'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
    ])
    def test_single_stage_forward_predict_mode(self, cfg_file):
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

        packed_inputs = demo_mm_inputs(2, [[3, 256, 256], [3, 255, 260]])
        batch_inputs, data_samples = detector.data_preprocessor(
            packed_inputs, False)
        # Test forward test
        detector.eval()
        with torch.no_grad():
            batch_results = detector.forward(
                batch_inputs, data_samples, mode='predict')
            self.assertEqual(len(batch_results), 2)
            self.assertIsInstance(batch_results[0], DetDataSample)

    @parameterized.expand([
        'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
        'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
    ])
    def test_single_stage_forward_tensor_mode(self, cfg_file):
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

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 125, 130]], with_mask=True)
        batch_inputs, data_samples = detector.data_preprocessor(
            packed_inputs, False)

        # TODO: Awaiting refactoring
        # out = detector.forward(batch_inputs, data_samples, mode='tensor')
        # self.assertIsInstance(out, tuple)
