# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized
from torch import Tensor

from mmdet.models.roi_heads.mask_heads import FusedSemanticHead


class TestFusedSemanticHead(TestCase):

    @parameterized.expand(['cpu', 'cuda'])
    def test_forward_loss(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        semantic_head = FusedSemanticHead(
            num_ins=5,
            fusion_level=1,
            in_channels=4,
            conv_out_channels=4,
            num_classes=6)
        feats = [
            torch.rand((1, 4, 32 // 2**(i + 1), 32 // 2**(i + 1)))
            for i in range(5)
        ]
        mask_pred, x = semantic_head(feats)
        labels = torch.randint(0, 6, (1, 1, 64, 64))
        loss = semantic_head.loss(mask_pred, labels)
        self.assertIsInstance(loss, Tensor)
