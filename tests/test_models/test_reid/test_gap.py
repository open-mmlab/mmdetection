# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet.models import GlobalAveragePooling


class TestGlobalAveragePooling(TestCase):

    def test_forward(self):
        inputs = torch.rand(32, 128, 14, 14)

        # test AdaptiveAvgPool2d
        neck = GlobalAveragePooling()
        outputs = neck(inputs)
        assert outputs.shape == (32, 128)

        # test kernel_size
        neck = GlobalAveragePooling(kernel_size=7)
        outputs = neck(inputs)
        assert outputs.shape == (32, 128 * 2 * 2)

        # test kenel_size and stride
        neck = GlobalAveragePooling(kernel_size=7, stride=2)
        outputs = neck(inputs)
        assert outputs.shape == (32, 128 * 4 * 4)
