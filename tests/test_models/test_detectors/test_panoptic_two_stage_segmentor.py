# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from parameterized import parameterized

from mmdet.core import DetDataSample
from mmdet.models import build_detector
from mmdet.testing._utils import demo_mm_inputs, get_detector_cfg
from mmdet.utils import register_all_modules

register_all_modules()


class TestTwoStagePanopticSegmentor(unittest.TestCase):

    def _create_model_cfg(self):
        cfg_file = 'panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py'
        model_cfg = get_detector_cfg(cfg_file)
        model_cfg.backbone.depth = 18
        model_cfg.neck.in_channels = [64, 128, 256, 512]
        model_cfg.backbone.init_cfg = None
        return model_cfg

    def test_init(self):
        model_cfg = self._create_model_cfg()
        detector = build_detector(model_cfg)
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
        detector = build_detector(model_cfg)

        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)

        packed_inputs = demo_mm_inputs(
            2,
            image_shapes=[(3, 128, 127), (3, 91, 92)],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=True)
        batch_inputs, data_samples = detector.data_preprocessor(
            packed_inputs, True)
        # Test loss mode
        losses = detector.forward(batch_inputs, data_samples, mode='loss')
        self.assertIsInstance(losses, dict)

    @parameterized.expand([('cpu', ), ('cuda', )])
    def test_forward_predict_mode(self, device):
        model_cfg = self._create_model_cfg()
        detector = build_detector(model_cfg)
        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)
        packed_inputs = demo_mm_inputs(
            2,
            image_shapes=[(3, 128, 127), (3, 91, 92)],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=True)
        batch_inputs, data_samples = detector.data_preprocessor(
            packed_inputs, False)
        # Test forward test
        detector.eval()
        with torch.no_grad():
            batch_results = detector.forward(
                batch_inputs, data_samples, mode='predict')
            self.assertEqual(len(batch_results), 2)
            self.assertIsInstance(batch_results[0], DetDataSample)

    @parameterized.expand([('cpu', ), ('cuda', )])
    def test_forward_tensor_mode(self, device):
        model_cfg = self._create_model_cfg()
        detector = build_detector(model_cfg)
        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 125, 130]],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=True)
        batch_inputs, data_samples = detector.data_preprocessor(
            packed_inputs, False)

        out = detector.forward(batch_inputs, data_samples, mode='tensor')
        self.assertIsInstance(out, tuple)

    @parameterized.expand([('cpu', ), ('cuda', )])
    def test_predict_mask(self, device):
        model_cfg = self._create_model_cfg()
        detector = build_detector(model_cfg)
        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 125, 130]],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=True)
        batch_inputs, batch_data_samples = detector.data_preprocessor(
            packed_inputs, False)
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        det_bboxes = [
            data_sample.gt_instances.bboxes
            for data_sample in batch_data_samples
        ]
        det_labels = [
            data_sample.gt_instances.labels
            for data_sample in batch_data_samples
        ]
        x = detector.extract_feat(batch_inputs)
        mask_results = detector._predict_mask(
            x, batch_img_metas, det_bboxes, det_labels, rescale=True)
        self.assertIsInstance(mask_results, dict)
        mask_results = detector._predict_mask(
            x, batch_img_metas, det_bboxes, det_labels, rescale=False)
        self.assertIsInstance(mask_results, dict)
