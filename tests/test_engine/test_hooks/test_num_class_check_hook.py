# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from unittest import TestCase
from unittest.mock import Mock

from mmcv.cnn import VGG
from mmengine.dataset import BaseDataset
from torch import nn

from mmdet.engine.hooks import NumClassCheckHook
from mmdet.models.roi_heads.mask_heads import FusedSemanticHead


class TestNumClassCheckHook(TestCase):

    def setUp(self):
        # Setup NumClassCheckHook
        hook = NumClassCheckHook()
        self.hook = hook

        # Setup runner mock
        runner = Mock()
        runner.model = Mock()
        runner.logger = Mock()
        runner.logger.warning = Mock()
        runner.train_dataloader = Mock()
        runner.val_dataloader = Mock()
        self.runner = runner

        # Setup dataset
        metainfo = dict(classes=None)
        self.none_classmeta_dataset = BaseDataset(
            metainfo=metainfo, lazy_init=True)
        metainfo = dict(classes='class_name')
        self.str_classmeta_dataset = BaseDataset(
            metainfo=metainfo, lazy_init=True)
        metainfo = dict(classes=('bus', 'car'))
        self.normal_classmeta_dataset = BaseDataset(
            metainfo=metainfo, lazy_init=True)

        # Setup valid model
        valid_model = nn.Module()
        valid_model.add_module('backbone', VGG(depth=11))
        fused_semantic_head = FusedSemanticHead(
            num_ins=1,
            fusion_level=0,
            num_convs=1,
            in_channels=1,
            conv_out_channels=1)
        valid_model.add_module('semantic_head', fused_semantic_head)
        rpn_head = nn.Module()
        rpn_head.num_classes = 1
        valid_model.add_module('rpn_head', rpn_head)
        bbox_head = nn.Module()
        bbox_head.num_classes = 2
        valid_model.add_module('bbox_head', bbox_head)
        self.valid_model = valid_model

        # Setup invalid model
        invalid_model = nn.Module()
        bbox_head = nn.Module()
        bbox_head.num_classes = 4
        invalid_model.add_module('bbox_head', bbox_head)
        self.invalid_model = invalid_model

    def test_before_train_epch(self):
        runner = deepcopy(self.runner)

        # Test when dataset.metainfo['classes'] is None
        runner.train_dataloader.dataset = self.none_classmeta_dataset
        self.hook.before_train_epoch(runner)
        runner.logger.warning.assert_called_once()
        # Test when dataset.metainfo['classes'] is a str
        runner.train_dataloader.dataset = self.str_classmeta_dataset
        with self.assertRaises(AssertionError):
            self.hook.before_train_epoch(runner)

        runner.train_dataloader.dataset = self.normal_classmeta_dataset
        # Test `num_classes` of model is compatible with dataset
        runner.model = self.valid_model
        self.hook.before_train_epoch(runner)
        # Test `num_classes` of model is not compatible with dataset
        runner.model = self.invalid_model
        with self.assertRaises(AssertionError):
            self.hook.before_train_epoch(runner)

    def test_before_val_epoch(self):
        runner = deepcopy(self.runner)

        # Test when dataset.metainfo['classes'] is None
        runner.val_dataloader.dataset = self.none_classmeta_dataset
        self.hook.before_val_epoch(runner)
        runner.logger.warning.assert_called_once()
        # Test when dataset.metainfo['classes'] is a str
        runner.val_dataloader.dataset = self.str_classmeta_dataset
        with self.assertRaises(AssertionError):
            self.hook.before_val_epoch(runner)

        runner.val_dataloader.dataset = self.normal_classmeta_dataset
        # Test `num_classes` of model is compatible with dataset
        runner.model = self.valid_model
        self.hook.before_val_epoch(runner)
        # Test `num_classes` of model is not compatible with dataset
        runner.model = self.invalid_model
        with self.assertRaises(AssertionError):
            self.hook.before_val_epoch(runner)
