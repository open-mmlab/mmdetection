import unittest

import torch

from mmdet.models.roi_heads.bbox_heads import SCNetBBoxHead


class TestSCNetBBoxHead(unittest.TestCase):

    def test_forward(self):
        x = torch.rand((2, 1, 16, 16))
        bbox_head = SCNetBBoxHead(
            num_shared_fcs=2,
            in_channels=1,
            roi_feat_size=16,
            conv_out_channels=1,
            fc_out_channels=256,
        )
        results = bbox_head(x, return_shared_feat=False)
        self.assertEqual(len(results), 2)
        results = bbox_head(x, return_shared_feat=True)
        self.assertEqual(len(results), 3)
