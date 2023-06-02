# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest
from unittest import TestCase

import torch
from mmengine.logging import MessageHub
from parameterized import parameterized

from mmdet.structures import DetDataSample
from mmdet.testing import demo_mm_inputs, get_detector_cfg
from mmdet.utils import register_all_modules


class TestSingleStageDetector(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand([
        'retinanet/retinanet_r18_fpn_1x_coco.py',
        'centernet/centernet_r18_8xb16-crop512-140e_coco.py',
        'fsaf/fsaf_r50_fpn_1x_coco.py',
        'yolox/yolox_tiny_8xb8-300e_coco.py',
        'yolo/yolov3_mobilenetv2_8xb24-320-300e_coco.py',
        'reppoints/reppoints-minmax_r50_fpn-gn_head-gn_1x_coco.py',
    ])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmdet.registry import MODELS
        detector = MODELS.build(model)
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.bbox_head)

    @parameterized.expand([
        ('retinanet/retinanet_r18_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('centernet/centernet_r18_8xb16-crop512-140e_coco.py', ('cpu',
                                                                'cuda')),
        ('fsaf/fsaf_r50_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('yolox/yolox_tiny_8xb8-300e_coco.py', ('cpu', 'cuda')),
        ('yolo/yolov3_mobilenetv2_8xb24-320-300e_coco.py', ('cpu', 'cuda')),
        ('reppoints/reppoints-minmax_r50_fpn-gn_head-gn_1x_coco.py', ('cpu',
                                                                      'cuda')),
    ])
    def test_single_stage_forward_loss_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_single_stage_forward_loss_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmdet.registry import MODELS
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = MODELS.build(model)
            detector.init_weights()

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
            data = detector.data_preprocessor(packed_inputs, True)
            losses = detector.forward(**data, mode='loss')
            self.assertIsInstance(losses, dict)

    @parameterized.expand([
        ('retinanet/retinanet_r18_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('centernet/centernet_r18_8xb16-crop512-140e_coco.py', ('cpu',
                                                                'cuda')),
        ('fsaf/fsaf_r50_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('yolox/yolox_tiny_8xb8-300e_coco.py', ('cpu', 'cuda')),
        ('yolo/yolov3_mobilenetv2_8xb24-320-300e_coco.py', ('cpu', 'cuda')),
        ('reppoints/reppoints-minmax_r50_fpn-gn_head-gn_1x_coco.py', ('cpu',
                                                                      'cuda')),
    ])
    def test_single_stage_forward_predict_mode(self, cfg_file, devices):
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

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
            data = detector.data_preprocessor(packed_inputs, False)
            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(**data, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], DetDataSample)

    @parameterized.expand([
        ('retinanet/retinanet_r18_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('centernet/centernet_r18_8xb16-crop512-140e_coco.py', ('cpu',
                                                                'cuda')),
        ('fsaf/fsaf_r50_fpn_1x_coco.py', ('cpu', 'cuda')),
        ('yolox/yolox_tiny_8xb8-300e_coco.py', ('cpu', 'cuda')),
        ('yolo/yolov3_mobilenetv2_8xb24-320-300e_coco.py', ('cpu', 'cuda')),
        ('reppoints/reppoints-minmax_r50_fpn-gn_head-gn_1x_coco.py', ('cpu',
                                                                      'cuda')),
    ])
    def test_single_stage_forward_tensor_mode(self, cfg_file, devices):
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

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
            data = detector.data_preprocessor(packed_inputs, False)
            batch_results = detector.forward(**data, mode='tensor')
            self.assertIsInstance(batch_results, tuple)
