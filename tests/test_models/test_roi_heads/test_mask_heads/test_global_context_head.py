# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized
from torch import Tensor

from mmdet.models.roi_heads.mask_heads import GlobalContextHead


class TestGlobalContextHead(TestCase):

    @parameterized.expand(['cpu', 'cuda'])
    def test_forward_loss(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        head = GlobalContextHead(
            num_convs=1, in_channels=4, conv_out_channels=4, num_classes=10)
        feats = [
            torch.rand((1, 4, 64 // 2**(i + 1), 64 // 2**(i + 1)))
            for i in range(5)
        ]
        mc_pred, x = head(feats)

        labels = [torch.randint(0, 10, (10, ))]
        loss = head.loss(mc_pred, labels)
        self.assertIsInstance(loss, Tensor)
