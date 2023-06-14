# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet.models import L2Loss


class TestL2Loss(TestCase):

    def test_l2_loss(self):
        pred = torch.Tensor([[1, 1, 0, 0, 0, 0, 1]])
        target = torch.Tensor([[1, 1, 0, 0, 0, 0, 0]])

        loss = L2Loss(
            neg_pos_ub=2,
            pos_margin=0,
            neg_margin=0.1,
            hard_mining=True,
            loss_weight=1.0)
        assert torch.allclose(loss(pred, target), torch.tensor(0.1350))
