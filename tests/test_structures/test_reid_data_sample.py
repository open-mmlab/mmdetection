# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import LabelData

from mmdet.structures import ReIDDataSample


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


class TestReIDDataSample(TestCase):

    def test_init(self):
        img_shape = (256, 128)
        ori_shape = (64, 64)
        num_classes = 5
        meta_info = dict(
            img_shape=img_shape, ori_shape=ori_shape, num_classes=num_classes)
        data_sample = ReIDDataSample(metainfo=meta_info)
        self.assertIn('img_shape', data_sample)
        self.assertIn('ori_shape', data_sample)
        self.assertIn('num_classes', data_sample)
        self.assertTrue(_equal(data_sample.get('img_shape'), img_shape))
        self.assertTrue(_equal(data_sample.get('ori_shape'), ori_shape))
        self.assertTrue(_equal(data_sample.get('num_classes'), num_classes))

    def test_set_gt_label(self):
        data_sample = ReIDDataSample(metainfo=dict(num_classes=5))
        method = getattr(data_sample, 'set_' + 'gt_label')

        # Test number
        method(1)
        label = data_sample.get('gt_label')
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test tensor with single number
        method(torch.tensor(2))
        label = data_sample.get('gt_label')
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test array with single number
        method(np.array(3))
        label = data_sample.get('gt_label')
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.LongTensor)

        # Test tensor
        _label = torch.tensor([1, 2, 3])
        method(_label)
        label = data_sample.get('gt_label')
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.Tensor)
        self.assertTrue(_equal(label.label, _label))

        # Test array
        _label = np.array([1, 2, 3])
        method(_label)
        label = data_sample.get('gt_label')
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.Tensor)
        self.assertTrue(_equal(label.label, torch.from_numpy(_label)))

        # Test Sequence
        _label = [1, 2, 3.]
        method(_label)
        label = data_sample.get('gt_label')
        self.assertIsInstance(label, LabelData)
        self.assertIsInstance(label.label, torch.Tensor)
        self.assertTrue(_equal(label.label, torch.tensor(_label)))

        # Test set num_classes
        self.assertEqual(label.num_classes, 5)

        # Test unavailable type
        with self.assertRaisesRegex(TypeError, "<class 'str'> is not"):
            method('hi')

    def test_set_gt_score(self):
        data_sample = ReIDDataSample(metainfo={'num_classes': 5})
        method = getattr(data_sample, 'set_' + 'gt_score')

        # Test set
        score = [0.1, 0.1, 0.6, 0.1, 0.1]
        method(torch.tensor(score))
        sample_gt_label = getattr(data_sample, 'gt_label')
        self.assertIn('score', sample_gt_label)
        torch.testing.assert_allclose(sample_gt_label.score, score)
        self.assertEqual(sample_gt_label.num_classes, 5)

        # Test set again
        score = [0.2, 0.1, 0.5, 0.1, 0.1]
        method(torch.tensor(score))
        torch.testing.assert_allclose(sample_gt_label.score, score)

        # Test invalid type
        with self.assertRaisesRegex(AssertionError, 'be a torch.Tensor'):
            method(score)

        # Test invalid dims
        with self.assertRaisesRegex(AssertionError, 'but got 2'):
            method(torch.tensor([score]))

        # Test invalid num_classes
        with self.assertRaisesRegex(AssertionError, r'length of value \(6\)'):
            method(torch.tensor(score + [0.1]))

        # Test auto inter num_classes
        data_sample = ReIDDataSample()
        method = getattr(data_sample, 'set_gt_score')
        method(torch.tensor(score))
        sample_gt_label = getattr(data_sample, 'gt_label')
        self.assertEqual(sample_gt_label.num_classes, len(score))

    def test_del_gt_label(self):
        data_sample = ReIDDataSample()
        self.assertNotIn('gt_label', data_sample)
        data_sample.set_gt_label(1)
        self.assertIn('gt_label', data_sample)
        del data_sample.gt_label
        self.assertNotIn('gt_label', data_sample)
