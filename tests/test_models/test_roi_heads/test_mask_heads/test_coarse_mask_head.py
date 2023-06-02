import unittest

import torch
from parameterized import parameterized

from mmdet.models.roi_heads.mask_heads import CoarseMaskHead


class TestCoarseMaskHead(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            CoarseMaskHead(num_fcs=0)

        with self.assertRaises(AssertionError):
            CoarseMaskHead(downsample_factor=0.5)

    @parameterized.expand(['cpu', 'cuda'])
    def test_forward(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        x = torch.rand((1, 32, 7, 7)).to(device)
        mask_head = CoarseMaskHead(
            downsample_factor=2,
            in_channels=32,
            conv_out_channels=32,
            roi_feat_size=7).to(device)
        mask_head.init_weights()
        res = mask_head(x)
        self.assertEqual(res.shape[-2:], (3, 3))

        mask_head = CoarseMaskHead(
            downsample_factor=1,
            in_channels=32,
            conv_out_channels=32,
            roi_feat_size=7).to(device)
        mask_head.init_weights()
        res = mask_head(x)
        self.assertEqual(res.shape[-2:], (7, 7))
