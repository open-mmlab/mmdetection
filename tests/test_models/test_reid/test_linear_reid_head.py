# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet.registry import MODELS
from mmdet.structures import ReIDDataSample
from mmdet.utils import register_all_modules


class TestLinearReIDHead(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        register_all_modules()
        head_cfg = dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=128,
            fc_channels=64,
            out_channels=32,
            num_classes=2,
            loss_cls=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
            loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU'))
        cls.head = MODELS.build(head_cfg)
        cls.inputs = (torch.rand(4, 128), torch.rand(4, 128))
        cls.data_samples = [
            ReIDDataSample().set_gt_label(label) for label in (0, 0, 1, 1)
        ]

    def test_forward(self):
        outputs = self.head(self.inputs)
        assert outputs.shape == (4, 32)

    def test_loss(self):
        losses = self.head.loss(self.inputs, self.data_samples)
        assert losses.keys() == {'triplet_loss', 'ce_loss', 'accuracy_top-1'}
        assert losses['ce_loss'].item() >= 0
        assert losses['triplet_loss'].item() >= 0

    def test_predict(self):
        predictions = self.head.predict(self.inputs, self.data_samples)
        for pred in predictions:
            assert isinstance(pred, ReIDDataSample)
            assert isinstance(pred.pred_feature, torch.Tensor)
            assert isinstance(pred.gt_label.label, torch.Tensor)
            assert pred.pred_feature.shape == (32, )
