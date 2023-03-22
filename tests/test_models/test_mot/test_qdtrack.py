# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest
from unittest import TestCase

import torch
from mmengine.logging import MessageHub
from mmengine.registry import init_default_scope
from parameterized import parameterized

from mmdet.registry import MODELS
from mmdet.testing import demo_track_inputs, get_detector_cfg


class TestQDTrack(TestCase):

    @classmethod
    def setUpClass(cls):
        init_default_scope('mmdet')

    @parameterized.expand([
        'qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17halftrain_'
        'test-mot17halfval.py',
    ])
    def test_qdtrack_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)

        model = MODELS.build(model)
        assert model.detector
        assert model.track_head

    @parameterized.expand([
        ('qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17'
         'halftrain_test-mot17halfval.py', ('cpu', 'cuda')),
    ])
    def test_qdtrack_forward_loss_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_qdtrack_forward_loss_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            _model = get_detector_cfg(cfg_file)
            # _scope_ will be popped after build
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = demo_track_inputs(
                batch_size=1,
                num_frames=2,
                key_frames_inds=[0],
                image_shapes=(3, 128, 128),
                num_items=None)
            out_data = model.data_preprocessor(packed_inputs, True)
            inputs, data_samples = out_data['inputs'], out_data['data_samples']
            # Test forward
            losses = model.forward(inputs, data_samples, mode='loss')
            assert isinstance(losses, dict)

    @parameterized.expand([
        ('qdtrack/qdtrack_faster-rcnn_r50_fpn_8xb2-4e_mot17'
         'halftrain_test-mot17halfval.py', ('cpu', 'cuda')),
    ])
    def test_qdtrack_forward_predict_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_bytetrack_forward_predict_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)

        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            _model = get_detector_cfg(cfg_file)
            model = MODELS.build(_model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                model = model.cuda()

            packed_inputs = demo_track_inputs(
                batch_size=1, num_frames=1, image_shapes=(3, 128, 128))
            out_data = model.data_preprocessor(packed_inputs, False)

            # Test forward test
            model.eval()
            with torch.no_grad():
                batch_results = model.forward(**out_data, mode='predict')
                assert len(batch_results) == 1
