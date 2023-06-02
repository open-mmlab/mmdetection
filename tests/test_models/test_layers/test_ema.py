# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.testing import assert_allclose

from mmdet.models.layers import ExpMomentumEMA


class TestEMA(TestCase):

    def test_exp_momentum_ema(self):
        model = nn.Sequential(nn.Conv2d(1, 5, kernel_size=3), nn.Linear(5, 10))
        # Test invalid gamma
        with self.assertRaisesRegex(AssertionError,
                                    'gamma must be greater than 0'):
            ExpMomentumEMA(model, gamma=-1)

        # Test EMA
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10))
        momentum = 0.1
        gamma = 4

        ema_model = ExpMomentumEMA(model, momentum=momentum, gamma=gamma)
        averaged_params = [
            torch.zeros_like(param) for param in model.parameters()
        ]
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            for p, p_avg in zip(model.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    m = (1 - momentum) * math.exp(-(1 + i) / gamma) + momentum
                    updated_averaged_params.append(
                        (p_avg * (1 - m) + p * m).clone())
            ema_model.update_parameters(model)
            averaged_params = updated_averaged_params

        for p_target, p_ema in zip(averaged_params, ema_model.parameters()):
            assert_allclose(p_target, p_ema)

    def test_exp_momentum_ema_update_buffer(self):
        model = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3), nn.BatchNorm2d(5, momentum=0.3),
            nn.Linear(5, 10))
        # Test invalid gamma
        with self.assertRaisesRegex(AssertionError,
                                    'gamma must be greater than 0'):
            ExpMomentumEMA(model, gamma=-1)

        # Test EMA with momentum annealing.
        momentum = 0.1
        gamma = 4

        ema_model = ExpMomentumEMA(
            model, gamma=gamma, momentum=momentum, update_buffers=True)
        averaged_params = [
            torch.zeros_like(param)
            for param in itertools.chain(model.parameters(), model.buffers())
            if param.size() != torch.Size([])
        ]
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            params = [
                param for param in itertools.chain(model.parameters(),
                                                   model.buffers())
                if param.size() != torch.Size([])
            ]
            for p, p_avg in zip(params, averaged_params):
                p.detach().add_(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    m = (1 - momentum) * math.exp(-(1 + i) / gamma) + momentum
                    updated_averaged_params.append(
                        (p_avg * (1 - m) + p * m).clone())
            ema_model.update_parameters(model)
            averaged_params = updated_averaged_params

        ema_params = [
            param for param in itertools.chain(ema_model.module.parameters(),
                                               ema_model.module.buffers())
            if param.size() != torch.Size([])
        ]
        for p_target, p_ema in zip(averaged_params, ema_params):
            assert_allclose(p_target, p_ema)
