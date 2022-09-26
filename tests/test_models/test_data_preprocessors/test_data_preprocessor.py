# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.logging import MessageHub

from mmdet.models.data_preprocessors import (BatchFixedSizePad,
                                             BatchSyncRandomResize,
                                             DetDataPreprocessor,
                                             MultiBranchDataPreprocessor)
from mmdet.structures import DetDataSample
from mmdet.testing import demo_mm_inputs


class TestDetDataPreprocessor(TestCase):

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

        data = {
            'inputs': [torch.randint(0, 256, (3, 11, 10))],
            'data_samples': [DetDataSample()]
        }
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']

        self.assertEqual(batch_inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(batch_data_samples), 1)

        # test channel_conversion
        processor = DetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(batch_data_samples), 1)

        # test padding
        data = {
            'inputs': [
                torch.randint(0, 256, (3, 10, 11)),
                torch.randint(0, 256, (3, 9, 14))
            ]
        }
        processor = DetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs.shape, (2, 3, 10, 14))
        self.assertIsNone(batch_data_samples)

        # test pad_size_divisor
        data = {
            'inputs': [
                torch.randint(0, 256, (3, 10, 11)),
                torch.randint(0, 256, (3, 9, 24))
            ],
            'data_samples': [DetDataSample()] * 2
        }
        processor = DetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], pad_size_divisor=5)
        out_data = processor(data)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs.shape, (2, 3, 10, 25))
        self.assertEqual(len(batch_data_samples), 2)
        for data_samples, expected_shape in zip(batch_data_samples,
                                                [(10, 15), (10, 25)]):
            self.assertEqual(data_samples.pad_shape, expected_shape)

        # test pad_mask=True and pad_seg=True
        processor = DetDataPreprocessor(
            pad_mask=True, mask_pad_value=0, pad_seg=True, seg_pad_value=0)
        packed_inputs = demo_mm_inputs(
            2, [[3, 10, 11], [3, 9, 24]],
            with_mask=True,
            with_semantic=True,
            use_box_type=True)
        packed_inputs['data_samples'][0].gt_sem_seg.sem_seg = torch.randint(
            0, 256, (1, 10, 11))
        packed_inputs['data_samples'][1].gt_sem_seg.sem_seg = torch.randint(
            0, 256, (1, 9, 24))
        mask_pad_sums = [
            x.gt_instances.masks.masks.sum()
            for x in packed_inputs['data_samples']
        ]
        seg_pad_sums = [
            x.gt_sem_seg.sem_seg.sum() for x in packed_inputs['data_samples']
        ]
        batch_data_samples = processor(
            packed_inputs, training=True)['data_samples']
        for data_samples, expected_shape, mask_pad_sum, seg_pad_sum in zip(
                batch_data_samples, [(10, 24), (10, 24)], mask_pad_sums,
                seg_pad_sums):
            self.assertEqual(data_samples.gt_instances.masks.masks.shape[-2:],
                             expected_shape)
            self.assertEqual(data_samples.gt_sem_seg.sem_seg.shape[-2:],
                             expected_shape)
            self.assertEqual(data_samples.gt_instances.masks.masks.sum(),
                             mask_pad_sum)
            self.assertEqual(data_samples.gt_sem_seg.sem_seg.sum(),
                             seg_pad_sum)

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
        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 128, 128]], use_box_type=True)
        batch_inputs = processor(packed_inputs, training=True)['inputs']
        self.assertEqual(batch_inputs.shape, (2, 3, 128, 128))

        # resize after one iter
        message_hub.update_info('iter', 1)
        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 128, 128]], use_box_type=True)
        batch_inputs = processor(packed_inputs, training=True)['inputs']
        self.assertEqual(batch_inputs.shape, (2, 3, 320, 320))

        packed_inputs = demo_mm_inputs(
            2, [[3, 128, 128], [3, 128, 128]], use_box_type=True)
        batch_inputs = processor(packed_inputs, training=False)['inputs']
        self.assertEqual(batch_inputs.shape, (2, 3, 128, 128))

    def test_batch_fixed_size_pad(self):
        # test pad_mask=False and pad_seg=False
        processor = DetDataPreprocessor(
            pad_mask=False,
            pad_seg=False,
            batch_augments=[
                dict(
                    type='BatchFixedSizePad',
                    size=(32, 32),
                    img_pad_value=0,
                    pad_mask=True,
                    mask_pad_value=0,
                    pad_seg=True,
                    seg_pad_value=0)
            ])
        self.assertTrue(
            isinstance(processor.batch_augments[0], BatchFixedSizePad))
        packed_inputs = demo_mm_inputs(
            2, [[3, 10, 11], [3, 9, 24]],
            with_mask=True,
            with_semantic=True,
            use_box_type=True)
        packed_inputs['data_samples'][0].gt_sem_seg.sem_seg = torch.randint(
            0, 256, (1, 10, 11))
        packed_inputs['data_samples'][1].gt_sem_seg.sem_seg = torch.randint(
            0, 256, (1, 9, 24))
        mask_pad_sums = [
            x.gt_instances.masks.masks.sum()
            for x in packed_inputs['data_samples']
        ]
        seg_pad_sums = [
            x.gt_sem_seg.sem_seg.sum() for x in packed_inputs['data_samples']
        ]
        data = processor(packed_inputs, training=True)
        batch_inputs, batch_data_samples = data['inputs'], data['data_samples']
        self.assertEqual(batch_inputs.shape[-2:], (32, 32))
        for data_samples, expected_shape, mask_pad_sum, seg_pad_sum in zip(
                batch_data_samples, [(32, 32), (32, 32)], mask_pad_sums,
                seg_pad_sums):
            self.assertEqual(data_samples.gt_instances.masks.masks.shape[-2:],
                             expected_shape)
            self.assertEqual(data_samples.gt_sem_seg.sem_seg.shape[-2:],
                             expected_shape)
            self.assertEqual(data_samples.gt_instances.masks.masks.sum(),
                             mask_pad_sum)
            self.assertEqual(data_samples.gt_sem_seg.sem_seg.sum(),
                             seg_pad_sum)

        # test pad_mask=True and pad_seg=True
        processor = DetDataPreprocessor(
            pad_mask=True,
            pad_seg=True,
            seg_pad_value=0,
            mask_pad_value=0,
            batch_augments=[
                dict(
                    type='BatchFixedSizePad',
                    size=(32, 32),
                    img_pad_value=0,
                    pad_mask=True,
                    mask_pad_value=0,
                    pad_seg=True,
                    seg_pad_value=0)
            ])
        self.assertTrue(
            isinstance(processor.batch_augments[0], BatchFixedSizePad))
        packed_inputs = demo_mm_inputs(
            2, [[3, 10, 11], [3, 9, 24]],
            with_mask=True,
            with_semantic=True,
            use_box_type=True)
        packed_inputs['data_samples'][0].gt_sem_seg.sem_seg = torch.randint(
            0, 256, (1, 10, 11))
        packed_inputs['data_samples'][1].gt_sem_seg.sem_seg = torch.randint(
            0, 256, (1, 9, 24))
        mask_pad_sums = [
            x.gt_instances.masks.masks.sum()
            for x in packed_inputs['data_samples']
        ]
        seg_pad_sums = [
            x.gt_sem_seg.sem_seg.sum() for x in packed_inputs['data_samples']
        ]
        data = processor(packed_inputs, training=True)
        batch_inputs, batch_data_samples = data['inputs'], data['data_samples']
        self.assertEqual(batch_inputs.shape[-2:], (32, 32))
        for data_samples, expected_shape, mask_pad_sum, seg_pad_sum in zip(
                batch_data_samples, [(32, 32), (32, 32)], mask_pad_sums,
                seg_pad_sums):
            self.assertEqual(data_samples.gt_instances.masks.masks.shape[-2:],
                             expected_shape)
            self.assertEqual(data_samples.gt_sem_seg.sem_seg.shape[-2:],
                             expected_shape)
            self.assertEqual(data_samples.gt_instances.masks.masks.sum(),
                             mask_pad_sum)
            self.assertEqual(data_samples.gt_sem_seg.sem_seg.sum(),
                             seg_pad_sum)

        # test negative pad/no pad
        processor = DetDataPreprocessor(
            pad_mask=True,
            pad_seg=True,
            seg_pad_value=0,
            mask_pad_value=0,
            batch_augments=[
                dict(
                    type='BatchFixedSizePad',
                    size=(5, 5),
                    img_pad_value=0,
                    pad_mask=True,
                    mask_pad_value=1,
                    pad_seg=True,
                    seg_pad_value=1)
            ])
        self.assertTrue(
            isinstance(processor.batch_augments[0], BatchFixedSizePad))
        packed_inputs = demo_mm_inputs(
            2, [[3, 10, 11], [3, 9, 24]],
            with_mask=True,
            with_semantic=True,
            use_box_type=True)
        packed_inputs['data_samples'][0].gt_sem_seg.sem_seg = torch.randint(
            0, 256, (1, 10, 11))
        packed_inputs['data_samples'][1].gt_sem_seg.sem_seg = torch.randint(
            0, 256, (1, 9, 24))
        mask_pad_sums = [
            x.gt_instances.masks.masks.sum()
            for x in packed_inputs['data_samples']
        ]
        seg_pad_sums = [
            x.gt_sem_seg.sem_seg.sum() for x in packed_inputs['data_samples']
        ]
        data = processor(packed_inputs, training=True)
        batch_inputs, batch_data_samples = data['inputs'], data['data_samples']
        self.assertEqual(batch_inputs.shape[-2:], (10, 24))
        for data_samples, expected_shape, mask_pad_sum, seg_pad_sum in zip(
                batch_data_samples, [(10, 24), (10, 24)], mask_pad_sums,
                seg_pad_sums):
            self.assertEqual(data_samples.gt_instances.masks.masks.shape[-2:],
                             expected_shape)
            self.assertEqual(data_samples.gt_sem_seg.sem_seg.shape[-2:],
                             expected_shape)
            self.assertEqual(data_samples.gt_instances.masks.masks.sum(),
                             mask_pad_sum)
            self.assertEqual(data_samples.gt_sem_seg.sem_seg.sum(),
                             seg_pad_sum)


class TestMultiBranchDataPreprocessor(TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.data_preprocessor = dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32)
        self.multi_data = {
            'inputs': {
                'sup': [torch.randint(0, 256, (3, 224, 224))],
                'unsup_teacher': [
                    torch.randint(0, 256, (3, 400, 600)),
                    torch.randint(0, 256, (3, 600, 400))
                ],
                'unsup_student': [
                    torch.randint(0, 256, (3, 700, 500)),
                    torch.randint(0, 256, (3, 500, 700))
                ]
            },
            'data_samples': {
                'sup': [DetDataSample()],
                'unsup_teacher': [DetDataSample(),
                                  DetDataSample()],
                'unsup_student': [DetDataSample(),
                                  DetDataSample()],
            }
        }
        self.data = {
            'inputs': [torch.randint(0, 256, (3, 224, 224))],
            'data_samples': [DetDataSample()]
        }

    def test_multi_data_preprocessor(self):
        processor = MultiBranchDataPreprocessor(self.data_preprocessor)
        # test processing multi_data when training
        multi_data = processor(self.multi_data, training=True)
        self.assertEqual(multi_data['inputs']['sup'].shape, (1, 3, 224, 224))
        self.assertEqual(multi_data['inputs']['unsup_teacher'].shape,
                         (2, 3, 608, 608))
        self.assertEqual(multi_data['inputs']['unsup_student'].shape,
                         (2, 3, 704, 704))
        self.assertEqual(len(multi_data['data_samples']['sup']), 1)
        self.assertEqual(len(multi_data['data_samples']['unsup_teacher']), 2)
        self.assertEqual(len(multi_data['data_samples']['unsup_student']), 2)
        # test processing data when testing
        data = processor(self.data)
        self.assertEqual(data['inputs'].shape, (1, 3, 224, 224))
        self.assertEqual(len(data['data_samples']), 1)
