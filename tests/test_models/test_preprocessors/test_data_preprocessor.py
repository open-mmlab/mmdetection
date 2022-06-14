# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.logging import MessageHub

from mmdet.core import DetDataSample
from mmdet.models.data_preprocessors import (BatchSyncRandomResize,
                                             DetDataPreprocessor)
from mmdet.testing import demo_mm_inputs


class TestClsDataPreprocessor(TestCase):

    def test_init(self):
        # test mean is None
        processor = DetDataPreprocessor()
        self.assertTrue(not hasattr(processor, 'mean'))
        self.assertTrue(processor._enable_normalize is False)

        # test mean is not None
        processor = DetDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])
        self.assertTrue(hasattr(processor, 'mean'))
        self.assertTrue(hasattr(processor, 'std'))
        self.assertTrue(processor._enable_normalize)

        # please specify both mean and std
        with self.assertRaises(AssertionError):
            DetDataPreprocessor(mean=[0, 0, 0])

        # bgr2rgb and rgb2bgr cannot be set to True at the same time
        with self.assertRaises(AssertionError):
            DetDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

    def test_forward(self):
        processor = DetDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        data = [{
            'inputs': torch.randint(0, 256, (3, 11, 10)),
            'data_sample': DetDataSample()
        }]
        inputs, data_samples = processor(data)
        print(inputs.dtype)
        self.assertEqual(inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(data_samples), 1)

        # test channel_conversion
        processor = DetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        inputs, data_samples = processor(data)
        self.assertEqual(inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(data_samples), 1)

        # test padding
        data = [{
            'inputs': torch.randint(0, 256, (3, 10, 11))
        }, {
            'inputs': torch.randint(0, 256, (3, 9, 14))
        }]
        processor = DetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        inputs, data_samples = processor(data)
        self.assertEqual(inputs.shape, (2, 3, 10, 14))
        self.assertIsNone(data_samples)

        # test pad_size_divisor
        data = [{
            'inputs': torch.randint(0, 256, (3, 10, 11)),
            'data_sample': DetDataSample()
        }, {
            'inputs': torch.randint(0, 256, (3, 9, 24)),
            'data_sample': DetDataSample()
        }]
        processor = DetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], pad_size_divisor=5)
        inputs, data_samples = processor(data)
        self.assertEqual(inputs.shape, (2, 3, 10, 25))
        self.assertEqual(len(data_samples), 2)
        for data_sample, expected_shape in zip(data_samples, [(10, 15),
                                                              (10, 25)]):
            self.assertEqual(data_sample.pad_shape, expected_shape)

    def test_batch_sync_random_resize(self):
        processor = DetDataPreprocessor(batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(320, 320),
                size_divisor=32,
                interval=1)
        ])
        self.assertTrue(
            isinstance(processor.batch_augments[0], BatchSyncRandomResize))
        message_hub = MessageHub.get_instance('test_batch_sync_random_resize')
        message_hub.update_info('iter', 0)
        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 128, 128]])
        batch_inputs, batch_data_samples = processor(
            packed_inputs, training=True)
        self.assertEqual(batch_inputs.shape, (2, 3, 128, 128))

        # resize after one iter
        message_hub.update_info('iter', 1)
        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 128, 128]])
        batch_inputs, batch_data_samples = processor(
            packed_inputs, training=True)
        self.assertEqual(batch_inputs.shape, (2, 3, 320, 320))

        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 128, 128]])
        batch_inputs, batch_data_samples = processor(
            packed_inputs, training=False)
        self.assertEqual(batch_inputs.shape, (2, 3, 128, 128))
