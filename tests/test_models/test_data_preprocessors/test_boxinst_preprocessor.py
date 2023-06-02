# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet.models.data_preprocessors import BoxInstDataPreprocessor
from mmdet.structures import DetDataSample
from mmdet.testing import demo_mm_inputs


class TestBoxInstDataPreprocessor(TestCase):

    def test_forward(self):
        processor = BoxInstDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        data = {
            'inputs': [torch.randint(0, 256, (3, 256, 256))],
            'data_samples': [DetDataSample()]
        }

        # Test evaluation mode
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']

        self.assertEqual(batch_inputs.shape, (1, 3, 256, 256))
        self.assertEqual(len(batch_data_samples), 1)

        # Test traning mode without gt bboxes
        packed_inputs = demo_mm_inputs(
            2, [[3, 256, 256], [3, 128, 128]], num_items=[0, 0])
        out_data = processor(packed_inputs, training=True)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']

        self.assertEqual(batch_inputs.shape, (2, 3, 256, 256))
        self.assertEqual(len(batch_data_samples), 2)
        self.assertEqual(len(batch_data_samples[0].gt_instances.masks), 0)
        self.assertEqual(
            len(batch_data_samples[0].gt_instances.pairwise_masks), 0)
        self.assertEqual(len(batch_data_samples[1].gt_instances.masks), 0)
        self.assertEqual(
            len(batch_data_samples[1].gt_instances.pairwise_masks), 0)

        # Test traning mode with gt bboxes
        packed_inputs = demo_mm_inputs(
            2, [[3, 256, 256], [3, 128, 128]], num_items=[2, 1])
        out_data = processor(packed_inputs, training=True)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']

        self.assertEqual(batch_inputs.shape, (2, 3, 256, 256))
        self.assertEqual(len(batch_data_samples), 2)
        self.assertEqual(len(batch_data_samples[0].gt_instances.masks), 2)
        self.assertEqual(
            len(batch_data_samples[0].gt_instances.pairwise_masks), 2)
        self.assertEqual(len(batch_data_samples[1].gt_instances.masks), 1)
        self.assertEqual(
            len(batch_data_samples[1].gt_instances.pairwise_masks), 1)
