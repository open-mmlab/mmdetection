# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest
from unittest import TestCase

import torch
from mmengine.logging import MessageHub
from mmengine.registry import init_default_scope
from parameterized import parameterized

from mmdet.registry import MODELS
from mmdet.testing import demo_mm_inputs, demo_track_inputs, get_detector_cfg


class TestByteTrack(TestCase):

    @classmethod
    def setUpClass(cls):
        init_default_scope('mmdet')

    @parameterized.expand([
        'ocsort/ocsort_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain'
        '_test-mot17halfval.py',
    ])
    def test_bytetrack_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        model.detector.neck.out_channels = 1
        model.detector.neck.num_csp_blocks = 1
        model.detector.bbox_head.in_channels = 1
        model.detector.bbox_head.feat_channels = 1
        model = MODELS.build(model)
        assert model.detector

    @parameterized.expand([
        ('ocsort/ocsort_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_'
         'test-mot17halfval.py', ('cpu', 'cuda')),
    ])
    def test_bytetrack_forward_loss_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_bytetrack_forward_loss_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            _model = get_detector_cfg(cfg_file)
            _model.detector.neck.out_channels = 1
            _model.detector.neck.num_csp_blocks = 1
            _model.detector.bbox_head.num_classes = 10
            _model.detector.bbox_head.in_channels = 1
            _model.detector.bbox_head.feat_channels = 1
            # _scope_ will be popped after build
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 125, 130]])
            data = model.data_preprocessor(packed_inputs, True)
            losses = model.forward(**data, mode='loss')
            assert isinstance(losses, dict)

    @parameterized.expand([
        ('ocsort/ocsort_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_'
         'test-mot17halfval.py', ('cpu', 'cuda')),
    ])
    def test_bytetrack_forward_predict_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_bytetrack_forward_predict_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)

        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            _model = get_detector_cfg(cfg_file)
            _model.detector.neck.out_channels = 1
            _model.detector.neck.num_csp_blocks = 1
            _model.detector.bbox_head.in_channels = 1
            _model.detector.bbox_head.feat_channels = 1
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = demo_track_inputs(
                batch_size=1,
                num_frames=2,
                image_shapes=[(3, 256, 256)],
                num_classes=1)
            out_data = model.data_preprocessor(packed_inputs, False)
            # Test forward test
            model.eval()
            with torch.no_grad():
                batch_results = model.forward(**out_data, mode='predict')
                assert len(batch_results) == 1
