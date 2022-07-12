import unittest

import torch

from mmdet.models.losses import GaussianFocalLoss


class TestGaussianFocalLoss(unittest.TestCase):

    def test_forward(self):
        pred = torch.rand((10, 4))
        target = torch.rand((10, 4))
        gaussian_focal_loss = GaussianFocalLoss()
        loss1 = gaussian_focal_loss(pred, target)
        self.assertIsInstance(loss1, torch.Tensor)

        loss2 = gaussian_focal_loss(pred, target, avg_factor=0.5)
        self.assertIsInstance(loss2, torch.Tensor)

        # test reduction
        gaussian_focal_loss = GaussianFocalLoss(reduction='none')
        loss = gaussian_focal_loss(pred, target)
        self.assertTrue(loss.shape == (10, 4))

        # test reduction_override
        loss = gaussian_focal_loss(pred, target, reduction_override='mean')
        self.assertTrue(loss.ndim == 0)

        # Only supports None, 'none', 'mean', 'sum'
        with self.assertRaises(AssertionError):
            gaussian_focal_loss(pred, target, reduction_override='max')

        # test pos_inds
        pos_inds = (torch.rand(5) * 8).long()
        pos_labels = (torch.rand(5) * 2).long()
        gaussian_focal_loss = GaussianFocalLoss()
        loss = gaussian_focal_loss(pred, target, pos_inds, pos_labels)
        self.assertIsInstance(loss, torch.Tensor)
