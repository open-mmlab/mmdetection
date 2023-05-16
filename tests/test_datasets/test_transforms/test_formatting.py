# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import numpy as np
import torch
from mmengine.structures import InstanceData, LabelData, PixelData

from mmdet.datasets.transforms import (PackDetInputs, PackReIDInputs,
                                       PackTrackInputs)
from mmdet.structures import DetDataSample, ReIDDataSample
from mmdet.structures.mask import BitmapMasks


class TestPackDetInputs(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = osp.join(osp.dirname(__file__), '../../data')
        img_path = osp.join(data_prefix, 'color.jpg')
        rng = np.random.RandomState(0)
        self.results1 = {
            'img_id': 1,
            'img_path': img_path,
            'ori_shape': (300, 400),
            'img_shape': (600, 800),
            'scale_factor': 2.0,
            'flip': False,
            'img': rng.rand(300, 400),
            'gt_seg_map': rng.rand(300, 400),
            'gt_masks':
            BitmapMasks(rng.rand(3, 300, 400), height=300, width=400),
            'gt_bboxes_labels': rng.rand(3, ),
            'gt_ignore_flags': np.array([0, 0, 1], dtype=bool),
            'proposals': rng.rand(2, 4),
            'proposals_scores': rng.rand(2, )
        }
        self.results2 = {
            'img_id': 1,
            'img_path': img_path,
            'ori_shape': (300, 400),
            'img_shape': (600, 800),
            'scale_factor': 2.0,
            'flip': False,
            'img': rng.rand(300, 400),
            'gt_seg_map': rng.rand(300, 400),
            'gt_masks':
            BitmapMasks(rng.rand(3, 300, 400), height=300, width=400),
            'gt_bboxes_labels': rng.rand(3, ),
            'proposals': rng.rand(2, 4),
            'proposals_scores': rng.rand(2, )
        }
        self.meta_keys = ('img_id', 'img_path', 'ori_shape', 'scale_factor',
                          'flip')

    def test_transform(self):
        transform = PackDetInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.results1))
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], DetDataSample)
        self.assertIsInstance(results['data_samples'].gt_instances,
                              InstanceData)
        self.assertIsInstance(results['data_samples'].ignored_instances,
                              InstanceData)
        self.assertEqual(len(results['data_samples'].gt_instances), 2)
        self.assertEqual(len(results['data_samples'].ignored_instances), 1)
        self.assertIsInstance(results['data_samples'].gt_sem_seg, PixelData)
        self.assertIsInstance(results['data_samples'].proposals, InstanceData)
        self.assertEqual(len(results['data_samples'].proposals), 2)
        self.assertIsInstance(results['data_samples'].proposals.bboxes,
                              torch.Tensor)
        self.assertIsInstance(results['data_samples'].proposals.scores,
                              torch.Tensor)

    def test_transform_without_ignore(self):
        transform = PackDetInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.results2))
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], DetDataSample)
        self.assertIsInstance(results['data_samples'].gt_instances,
                              InstanceData)
        self.assertIsInstance(results['data_samples'].ignored_instances,
                              InstanceData)
        self.assertEqual(len(results['data_samples'].gt_instances), 3)
        self.assertEqual(len(results['data_samples'].ignored_instances), 0)
        self.assertIsInstance(results['data_samples'].gt_sem_seg, PixelData)
        self.assertIsInstance(results['data_samples'].proposals, InstanceData)
        self.assertEqual(len(results['data_samples'].proposals), 2)
        self.assertIsInstance(results['data_samples'].proposals.bboxes,
                              torch.Tensor)
        self.assertIsInstance(results['data_samples'].proposals.scores,
                              torch.Tensor)

    def test_repr(self):
        transform = PackDetInputs(meta_keys=self.meta_keys)
        self.assertEqual(
            repr(transform), f'PackDetInputs(meta_keys={self.meta_keys})')


class TestPackTrackInputs(unittest.TestCase):

    def setUp(self):
        self.H, self.W = 5, 10
        self.img = np.zeros((self.H, self.W, 3))
        self.gt_bboxes = np.zeros((2, 4))
        self.gt_masks = BitmapMasks(
            np.random.rand(2, self.H, self.W), height=self.H, width=self.W)
        self.gt_bboxes_labels = [
            np.zeros((2, )),
            np.zeros((2, )) + 1,
            np.zeros((2, )) - 1
        ]
        self.gt_instances_ids = [
            np.ones((2, ), dtype=np.int32),
            np.ones((2, ), dtype=np.int32) - 1,
            np.ones((2, ), dtype=np.int32) + 1
        ]
        self.frame_id = [0, 1, 2]
        self.scale_factor = [1.0, 1.5, 2.0]
        self.flip = [False] * 3
        self.ori_shape = [(self.H, self.W)] * 3
        self.img_id = [0, 1, 2]
        self.results_1 = dict(
            img=[self.img.copy(),
                 self.img.copy(),
                 self.img.copy()],
            gt_bboxes=[
                self.gt_bboxes.copy(),
                self.gt_bboxes.copy(),
                self.gt_bboxes.copy()
            ],
            gt_bboxes_labels=copy.deepcopy(self.gt_bboxes_labels),
            gt_instances_ids=copy.deepcopy(self.gt_instances_ids),
            gt_masks=[
                copy.deepcopy(self.gt_masks),
                copy.deepcopy(self.gt_masks),
                copy.deepcopy(self.gt_masks)
            ],
            frame_id=self.frame_id,
            ori_shape=self.ori_shape,
            scale_factor=self.scale_factor,
            flip=self.flip,
            img_id=self.img_id,
            key_frame_flags=[False, True, False])

        self.results_2 = copy.deepcopy(self.results_1)
        self.gt_ignore_flags = [
            np.array([0, 1], dtype=np.bool),
            np.array([1, 0], dtype=np.bool),
            np.array([0, 0], dtype=np.bool)
        ]
        self.results_2.update(
            dict(gt_ignore_flags=copy.deepcopy(self.gt_ignore_flags)))

        self.meta_keys = ('frame_id', 'ori_shape', 'scale_factor', 'flip')
        self.pack_track_inputs = PackTrackInputs(meta_keys=self.meta_keys)

    def test_transform_without_ignore(self):
        track_results = self.pack_track_inputs(self.results_1)
        assert isinstance(track_results, dict)

        inputs = track_results['inputs']
        assert isinstance(inputs, torch.Tensor)
        assert inputs.shape == (3, 3, self.H, self.W)

        track_data_sample = track_results['data_samples']
        assert len(track_data_sample) == 3
        assert 'key_frames_inds' in track_data_sample.metainfo and \
            track_data_sample.key_frames_inds == [1]
        assert 'ref_frames_inds' in track_data_sample.metainfo and \
            track_data_sample.ref_frames_inds == [0, 2]
        for i, data_sample in enumerate(track_data_sample):
            assert data_sample.gt_instances.bboxes.shape == (2, 4)
            assert len(data_sample.gt_instances.masks) == 2
            assert (data_sample.gt_instances.labels.numpy() ==
                    self.gt_bboxes_labels[i]).all()
            assert (data_sample.gt_instances.instances_ids.numpy() ==
                    self.gt_instances_ids[i]).all()
            for key in self.meta_keys:
                assert data_sample.metainfo[key] == getattr(self, key)[i]

    def test_transform_with_ignore(self):
        track_results = self.pack_track_inputs(self.results_2)
        assert isinstance(track_results, dict)

        inputs = track_results['inputs']
        assert isinstance(inputs, torch.Tensor)
        assert inputs.shape == (3, 3, self.H, self.W)

        track_data_sample = track_results['data_samples']
        assert len(track_data_sample) == 3
        for i, data_sample in enumerate(track_data_sample):
            valid_mask = ~self.gt_ignore_flags[i]
            valid_len = valid_mask.sum().item()
            assert data_sample.gt_instances.bboxes.shape == (valid_len, 4)
            assert len(data_sample.gt_instances.masks) == valid_len
            assert (data_sample.gt_instances.labels.numpy() ==
                    self.gt_bboxes_labels[i][valid_mask]).all()
            assert (data_sample.gt_instances.instances_ids.numpy() ==
                    self.gt_instances_ids[i][valid_mask]).all()
            for key in self.meta_keys:
                assert data_sample.metainfo[key] == getattr(self, key)[i]


class TestPackReIDInputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = dict(
            img=np.random.randn(256, 128, 3),
            gt_label=0,
            img_path='',
            ori_shape=(128, 128),
            img_shape=(256, 128),
            scale=(128, 256),
            scale_factor=(1., 2.),
            flip=False,
            flip_direction=None)
        cls.pack_reid_inputs = PackReIDInputs(
            meta_keys=('flip', 'flip_direction'))

    def test_transform(self):
        results = self.pack_reid_inputs(self.results)
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        data_sample = results['data_samples']
        self.assertIsInstance(data_sample, ReIDDataSample)
        self.assertIsInstance(data_sample.gt_label, LabelData)
        self.assertEqual(data_sample.img_path, '')
        self.assertEqual(data_sample.ori_shape, (128, 128))
        self.assertEqual(data_sample.img_shape, (256, 128))
        self.assertEqual(data_sample.scale, (128, 256))
        self.assertEqual(data_sample.scale_factor, (1., 2.))
        self.assertEqual(data_sample.flip, False)
        self.assertIsNone(data_sample.flip_direction)

    def test_repr(self):
        self.assertEqual(
            repr(self.pack_reid_inputs),
            f'PackReIDInputs(meta_keys={self.pack_reid_inputs.meta_keys})')
