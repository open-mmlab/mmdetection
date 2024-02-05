# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.device import is_musa_available
from parameterized import parameterized
from torch import Tensor

from mmdet.models.roi_heads.mask_heads import FeatureRelayHead


class TestFeatureRelayHead(TestCase):

    @parameterized.expand(['cpu', 'cuda', 'musa'])
    def test_forward(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')
        if device == 'musa':
            if is_musa_available():
                return unittest.skip('test requires GPU and torch+musa')
        mask_head = FeatureRelayHead(in_channels=10, out_conv_channels=10)

        x = torch.rand((1, 10))
        results = mask_head(x)
        self.assertIsInstance(results, Tensor)
        x = torch.empty((0, 10))
        results = mask_head(x)
        self.assertEqual(results, None)
