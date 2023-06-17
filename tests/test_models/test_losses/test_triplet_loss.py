# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet.models import TripletLoss


class TestTripletLoss(TestCase):

    def test_triplet_loss(self):
        feature = torch.Tensor([[1, 1], [1, 1], [0, 0], [0, 0]])
        label = torch.Tensor([1, 1, 0, 0])

        loss = TripletLoss(margin=0.3, loss_weight=1.0)
        assert torch.allclose(loss(feature, label), torch.tensor(0.))

        label = torch.Tensor([1, 0, 1, 0])
        assert torch.allclose(loss(feature, label), torch.tensor(1.7142))
