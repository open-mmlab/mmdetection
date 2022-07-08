# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet.models.roi_heads.bbox_heads import DoubleConvFCBBoxHead


class TestDoubleBboxHead(TestCase):

    @parameterized.expand(['cpu', 'cuda'])
    def test_forward_loss(self, device):
        if device == 'cuda':
            if not torch.cuda.is_available():
                return unittest.skip('test requires GPU and torch+cuda')

        double_bbox_head = DoubleConvFCBBoxHead(
            num_convs=4,
            num_fcs=2,
            in_channels=1,
            conv_out_channels=4,
            fc_out_channels=4)
        double_bbox_head = double_bbox_head.to(device=device)

        num_samples = 4
        feats = torch.rand((num_samples, 1, 7, 7)).to(device)
        double_bbox_head(x_cls=feats, x_reg=feats)
