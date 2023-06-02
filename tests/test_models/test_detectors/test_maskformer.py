# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from parameterized import parameterized

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.testing._utils import demo_mm_inputs, get_detector_cfg
from mmdet.utils import register_all_modules


class TestMaskFormer(unittest.TestCase):

    def setUp(self):
        register_all_modules()

    def _create_model_cfg(self):
        cfg_path = 'maskformer/maskformer_r50_ms-16xb1-75e_coco.py'
        model_cfg = get_detector_cfg(cfg_path)
        base_channels = 32
        model_cfg.backbone.depth = 18
        model_cfg.backbone.init_cfg = None
        model_cfg.backbone.base_channels = base_channels
        model_cfg.panoptic_head.in_channels = [
            base_channels * 2**i for i in range(4)
        ]
        model_cfg.panoptic_head.feat_channels = base_channels
        model_cfg.panoptic_head.out_channels = base_channels
        model_cfg.panoptic_head.pixel_decoder.encoder.\
            layer_cfg.self_attn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.pixel_decoder.encoder.\
            layer_cfg.ffn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.pixel_decoder.encoder.\
            layer_cfg.ffn_cfg.feedforward_channels = base_channels * 8
        model_cfg.panoptic_head.pixel_decoder.\
            positional_encoding.num_feats = base_channels // 2
        model_cfg.panoptic_head.positional_encoding.\
            num_feats = base_channels // 2
        model_cfg.panoptic_head.transformer_decoder.\
            layer_cfg.self_attn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.transformer_decoder. \
            layer_cfg.cross_attn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.transformer_decoder.\
            layer_cfg.ffn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.transformer_decoder.\
            layer_cfg.ffn_cfg.feedforward_channels = base_channels * 8
        return model_cfg

    def test_init(self):
        model_cfg = self._create_model_cfg()
        detector = MODELS.build(model_cfg)
        detector.init_weights()
        assert detector.backbone
        assert detector.panoptic_head

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


class TestMask2Former(unittest.TestCase):

    def setUp(self):
        register_all_modules()

    def _create_model_cfg(self, cfg_path):
        model_cfg = get_detector_cfg(cfg_path)
        base_channels = 32
        model_cfg.backbone.depth = 18
        model_cfg.backbone.init_cfg = None
        model_cfg.backbone.base_channels = base_channels
        model_cfg.panoptic_head.in_channels = [
            base_channels * 2**i for i in range(4)
        ]
        model_cfg.panoptic_head.feat_channels = base_channels
        model_cfg.panoptic_head.out_channels = base_channels
        model_cfg.panoptic_head.pixel_decoder.encoder.\
            layer_cfg.self_attn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.pixel_decoder.encoder.\
            layer_cfg.ffn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.pixel_decoder.encoder.\
            layer_cfg.ffn_cfg.feedforward_channels = base_channels * 4
        model_cfg.panoptic_head.pixel_decoder.\
            positional_encoding.num_feats = base_channels // 2
        model_cfg.panoptic_head.positional_encoding.\
            num_feats = base_channels // 2
        model_cfg.panoptic_head.transformer_decoder.\
            layer_cfg.self_attn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.transformer_decoder. \
            layer_cfg.cross_attn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.transformer_decoder.\
            layer_cfg.ffn_cfg.embed_dims = base_channels
        model_cfg.panoptic_head.transformer_decoder.\
            layer_cfg.ffn_cfg.feedforward_channels = base_channels * 8

        return model_cfg

    def test_init(self):
        model_cfg = self._create_model_cfg(
            'mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py')
        detector = MODELS.build(model_cfg)
        detector.init_weights()
        assert detector.backbone
        assert detector.panoptic_head

    @parameterized.expand([
        ('cpu', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'),
        ('cpu', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'),
        ('cuda', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'),
        ('cuda', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco.py')
    ])
    def test_forward_loss_mode(self, device, cfg_path):
        print(device, cfg_path)
        with_semantic = 'panoptic' in cfg_path
        model_cfg = self._create_model_cfg(cfg_path)
        detector = MODELS.build(model_cfg)

        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)

        packed_inputs = demo_mm_inputs(
            2,
            image_shapes=[(3, 128, 127), (3, 91, 92)],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=with_semantic)
        data = detector.data_preprocessor(packed_inputs, True)
        # Test loss mode
        losses = detector.forward(**data, mode='loss')
        self.assertIsInstance(losses, dict)

    @parameterized.expand([
        ('cpu', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'),
        ('cpu', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'),
        ('cuda', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'),
        ('cuda', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco.py')
    ])
    def test_forward_predict_mode(self, device, cfg_path):
        with_semantic = 'panoptic' in cfg_path
        model_cfg = self._create_model_cfg(cfg_path)
        detector = MODELS.build(model_cfg)
        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)
        packed_inputs = demo_mm_inputs(
            2,
            image_shapes=[(3, 128, 127), (3, 91, 92)],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=with_semantic)
        data = detector.data_preprocessor(packed_inputs, False)
        # Test forward test
        detector.eval()
        with torch.no_grad():
            batch_results = detector.forward(**data, mode='predict')
            self.assertEqual(len(batch_results), 2)
            self.assertIsInstance(batch_results[0], DetDataSample)

    @parameterized.expand([
        ('cpu', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'),
        ('cpu', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'),
        ('cuda', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'),
        ('cuda', 'mask2former/mask2former_r50_8xb2-lsj-50e_coco.py')
    ])
    def test_forward_tensor_mode(self, device, cfg_path):
        with_semantic = 'panoptic' in cfg_path
        model_cfg = self._create_model_cfg(cfg_path)
        detector = MODELS.build(model_cfg)
        if device == 'cuda' and not torch.cuda.is_available():
            return unittest.skip('test requires GPU and torch+cuda')
        detector = detector.to(device)

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 125, 130]],
            sem_seg_output_strides=1,
            with_mask=True,
            with_semantic=with_semantic)
        data = detector.data_preprocessor(packed_inputs, False)
        out = detector.forward(**data, mode='tensor')
        self.assertIsInstance(out, tuple)
