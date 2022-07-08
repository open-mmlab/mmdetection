# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized
from torch import Tensor

from mmdet.models.roi_heads.mask_heads import HTCMaskHead


class TestHTCMaskHead(TestCase):

    @parameterized.expand(['cpu', 'cuda'])
    def test_forward(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')
        num_classes = 6
        mask_head = HTCMaskHead(
            with_conv_res=True,
            num_convs=1,
            in_channels=1,
            conv_out_channels=1,
            num_classes=num_classes)

        x = torch.rand((1, 1, 10, 10))
        res_feat = torch.rand((1, 1, 10, 10))

        with self.assertRaises(AssertionError):
            mask_head(x, return_logits=False, return_feat=False)

        results = mask_head(x)
        self.assertEqual(len(results), 2)
        results = mask_head(x, res_feat=res_feat)
        self.assertEqual(len(results), 2)

        results = mask_head(x, return_logits=False)
        self.assertIsInstance(results, Tensor)
        results = mask_head(x, return_feat=False)
        self.assertIsInstance(results, Tensor)

        results = mask_head(x, res_feat=res_feat, return_logits=False)
        self.assertIsInstance(results, Tensor)
        results = mask_head(x, res_feat=res_feat, return_feat=False)
        self.assertIsInstance(results, Tensor)
