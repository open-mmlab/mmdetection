# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmdet.models.necks import CTResNetNeck


class TestCTResNetNeck(unittest.TestCase):

    def test_init(self):
        # num_filters/num_kernels must be same length
        with self.assertRaises(AssertionError):
            CTResNetNeck(
                in_channels=10,
                num_deconv_filters=(10, 10),
                num_deconv_kernels=(4, ))

        ct_resnet_neck = CTResNetNeck(
            in_channels=16,
            num_deconv_filters=(8, 8),
            num_deconv_kernels=(4, 4),
            use_dcn=False)
        ct_resnet_neck.init_weights()

    def test_forward(self):
        in_channels = 16
        num_filters = (8, 8)
        num_kernels = (4, 4)
        feat = torch.rand(1, 16, 4, 4)
        ct_resnet_neck = CTResNetNeck(
            in_channels=in_channels,
            num_deconv_filters=num_filters,
            num_deconv_kernels=num_kernels,
            use_dcn=False)

        # feat must be list or tuple
        with self.assertRaises(AssertionError):
            ct_resnet_neck(feat)

        out_feat = ct_resnet_neck([feat])[0]
        self.assertEqual(out_feat.shape, (1, num_filters[-1], 16, 16))

        if torch.cuda.is_available():
            # test dcn
            ct_resnet_neck = CTResNetNeck(
                in_channels=in_channels,
                num_deconv_filters=num_filters,
                num_deconv_kernels=num_kernels)
            ct_resnet_neck = ct_resnet_neck.cuda()
            feat = feat.cuda()
            out_feat = ct_resnet_neck([feat])[0]
            self.assertEqual(out_feat.shape, (1, num_filters[-1], 16, 16))
