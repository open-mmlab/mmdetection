# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmdet.models.data_preprocessors import BatchResize, DetDataPreprocessor
from mmdet.testing import demo_mm_inputs


class TestDetDataPreprocessor(TestCase):

    def test_batch_resize(self):

        processor = DetDataPreprocessor(
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            bgr_to_rgb=False,
            batch_augments=[
                dict(type='BatchResize', scale=(32, 32), pad_size_divisor=32)
            ])
        self.assertTrue(isinstance(processor.batch_augments[0], BatchResize))

        packed_inputs = demo_mm_inputs(
            2, [[3, 10, 11], [3, 9, 24]], use_box_type=True)
        data = processor(packed_inputs, training=True)
        batch_inputs, batch_data_samples = data['inputs'], data['data_samples']
        self.assertEqual(batch_inputs.shape[-2:], (32, 32))
        self.assertEqual(batch_data_samples[0].scale_factor,
                         batch_data_samples[1].scale_factor)
