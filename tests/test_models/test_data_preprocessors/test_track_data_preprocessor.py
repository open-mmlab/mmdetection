# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmdet.models.data_preprocessors import TrackDataPreprocessor
from mmdet.testing import demo_track_inputs


class TestTrackDataPreprocessor(TestCase):

    def test_init(self):
        # test mean is None
        processor = TrackDataPreprocessor()
        self.assertTrue(not hasattr(processor, 'mean'))
        self.assertTrue(processor._enable_normalize is False)

        # test mean is not None
        processor = TrackDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])
        self.assertTrue(hasattr(processor, 'mean'))
        self.assertTrue(hasattr(processor, 'std'))
        self.assertTrue(processor._enable_normalize)

        # please specify both mean and std
        with self.assertRaises(AssertionError):
            TrackDataPreprocessor(mean=[0, 0, 0])

        # bgr2rgb and rgb2bgr cannot be set to True at the same time
        with self.assertRaises(AssertionError):
            TrackDataPreprocessor(bgr_to_rgb=True, rgb_to_bgr=True)

    def test_forward(self):
        processor = TrackDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        data = demo_track_inputs(
            batch_size=1,
            num_frames=1,
            image_shapes=(3, 11, 10),
            num_items=[1])
        out_data = processor(data)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        self.assertEqual(inputs.shape, (1, 1, 3, 11, 10))
        self.assertEqual(len(data_samples), 1)

        # test channel_conversion
        processor = TrackDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        self.assertEqual(len(data_samples), 1)

        # test padding
        data = demo_track_inputs(
            batch_size=2,
            num_frames=2,
            image_shapes=[(3, 10, 11), (3, 9, 14)],
            num_items=[1, 1])
        out_data = processor(data)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        self.assertEqual(inputs.shape, (2, 2, 3, 10, 14))

        # test pad_size_divisor
        data = demo_track_inputs(
            batch_size=2,
            num_frames=2,
            image_shapes=[(3, 10, 11), (3, 9, 24)],
            num_items=[1, 1])
        processor = TrackDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], pad_size_divisor=5)
        out_data = processor(data)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        self.assertEqual(inputs.shape, (2, 2, 3, 10, 25))
        self.assertEqual(len(data_samples), 2)
        for track_data_sample, expected_shape in zip(data_samples, [(10, 15),
                                                                    (10, 25)]):
            for det_data_sample in track_data_sample.video_data_samples:
                self.assertEqual(det_data_sample.pad_shape, expected_shape)

        # test pad_mask=True
        data = demo_track_inputs(
            batch_size=2,
            num_frames=2,
            image_shapes=[(3, 10, 11), (3, 9, 24)],
            num_items=[1, 1],
            with_mask=True)
        processor = TrackDataPreprocessor(pad_mask=True, mask_pad_value=0)
        mask_pad_sums = []
        for track_data_sample in data['data_samples']:
            pad_sum_per_sample = []
            for x in track_data_sample.video_data_samples:
                pad_sum_per_sample.append(x.gt_instances.masks.masks.sum())
            mask_pad_sums.append(pad_sum_per_sample)
        out_data = processor(data, training=True)
        inputs, data_samples = out_data['inputs'], out_data['data_samples']
        for track_data_sample, expected_shape, mask_pad_sum in zip(
                data_samples, [(10, 24), (10, 24)], mask_pad_sums):
            for i, det_data_sample in enumerate(
                    track_data_sample.video_data_samples):
                self.assertEqual(
                    det_data_sample.gt_instances.masks.masks.shape[-2:],
                    expected_shape)
                self.assertEqual(
                    det_data_sample.gt_instances.masks.masks.sum(),
                    mask_pad_sum[i])
