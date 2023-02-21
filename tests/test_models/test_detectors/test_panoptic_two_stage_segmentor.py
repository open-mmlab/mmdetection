# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from parameterized import parameterized

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.testing._utils import demo_mm_inputs, get_detector_cfg
from mmdet.utils import register_all_modules


class TestTwoStagePanopticSegmentor(unittest.TestCase):

    def setUp(self):
        register_all_modules()

    def _create_model_cfg(self):
        cfg_file = 'panoptic_fpn/panoptic-fpn_r50_fpn_1x_coco.py'
        model_cfg = get_detector_cfg(cfg_file)
        model_cfg.backbone.depth = 18
        model_cfg.neck.in_channels = [64, 128, 256, 512]
        model_cfg.backbone.init_cfg = None
        return model_cfg

    def test_init(self):
        model_cfg = self._create_model_cfg()
        detector = MODELS.build(model_cfg)
        assert detector.backbone
        assert detector.neck
        assert detector.rpn_head
        assert detector.roi_head
        assert detector.roi_head.mask_head
        assert detector.with_semantic_head
        assert detector.with_panoptic_fusion_head

    @parameterized.expand([('cpu', ), ('cuda', )])
    def test_forward_loss_mode(self, device):
        model_cfg = self._create_model_cfg()
        detector = MODELS.build(model_cfg)

        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)

        packed_inputs = demo_mm_inputs(
            2,
            image_shapes=[(3, 128, 127), (3, 91, 92)],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=True)
        data = detector.data_preprocessor(packed_inputs, True)
        # Test loss mode
        losses = detector.forward(**data, mode='loss')
        self.assertIsInstance(losses, dict)

    @parameterized.expand([('cpu', ), ('cuda', )])
    def test_forward_predict_mode(self, device):
        model_cfg = self._create_model_cfg()
        detector = MODELS.build(model_cfg)
        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)
        packed_inputs = demo_mm_inputs(
            2,
            image_shapes=[(3, 128, 127), (3, 91, 92)],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=True)
        data = detector.data_preprocessor(packed_inputs, False)
        # Test forward test
        detector.eval()
        with torch.no_grad():
            batch_results = detector.forward(**data, mode='predict')
            self.assertEqual(len(batch_results), 2)
            self.assertIsInstance(batch_results[0], DetDataSample)

    @parameterized.expand([('cpu', ), ('cuda', )])
    def test_forward_tensor_mode(self, device):
        model_cfg = self._create_model_cfg()
        detector = MODELS.build(model_cfg)
        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 125, 130]],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=True)
        data = detector.data_preprocessor(packed_inputs, False)
        out = detector.forward(**data, mode='tensor')
        self.assertIsInstance(out, tuple)
