# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet.models import FcModule


class TestFcModule(TestCase):

    def test_forward(self):
        inputs = torch.rand(32, 128)

        # test
        fc = FcModule(
            in_channels=128,
            out_channels=32,
        )
        fc.init_weights()
        outputs = fc(inputs)
        assert outputs.shape == (32, 32)

        # test with norm
        fc = FcModule(
            in_channels=128,
            out_channels=32,
            norm_cfg=dict(type='BN1d'),
        )
        outputs = fc(inputs)
        assert outputs.shape == (32, 32)

        # test with norm and act
        fc = FcModule(
            in_channels=128,
            out_channels=32,
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'),
        )
        outputs = fc(inputs)
        assert outputs.shape == (32, 32)
