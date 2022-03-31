# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.core.utils import LearningRateDecayOptimizerConstructor

base_lr = 1
decay_rate = 2
base_wd = 0.05
weight_decay = 0.05

stage_wise_gt_lst = [{
    'weight_decay': 0.0,
    'lr_scale': 128
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}, {
    'weight_decay': 0.05,
    'lr_scale': 64
}, {
    'weight_decay': 0.0,
    'lr_scale': 64
}, {
    'weight_decay': 0.05,
    'lr_scale': 32
}, {
    'weight_decay': 0.0,
    'lr_scale': 32
}, {
    'weight_decay': 0.05,
    'lr_scale': 16
}, {
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 8
}, {
    'weight_decay': 0.0,
    'lr_scale': 8
}, {
    'weight_decay': 0.05,
    'lr_scale': 128
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}]

layer_wise_gt_lst = [{
    'weight_decay': 0.0,
    'lr_scale': 128
}, {
    'weight_decay': 0.0,
    'lr_scale': 1
}, {
    'weight_decay': 0.05,
    'lr_scale': 64
}, {
    'weight_decay': 0.0,
    'lr_scale': 64
}, {
    'weight_decay': 0.05,
    'lr_scale': 32
}, {
    'weight_decay': 0.0,
    'lr_scale': 32
}, {
    'weight_decay': 0.05,
    'lr_scale': 16
}, {
    'weight_decay': 0.0,
    'lr_scale': 16
}, {
    'weight_decay': 0.05,
    'lr_scale': 2
}, {
    'weight_decay': 0.0,
    'lr_scale': 2
}, {
    'weight_decay': 0.05,
    'lr_scale': 128
}, {
    'weight_decay': 0.05,
    'lr_scale': 1
}]


class ConvNeXtExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleList()
        self.backbone.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(ConvModule(3, 4, kernel_size=1, bias=True))
            self.backbone.stages.append(stage)
        self.backbone.norm0 = nn.BatchNorm2d(2)

        # add some variables to meet unit test coverate rate
        self.backbone.cls_token = nn.Parameter(torch.ones(1))
        self.backbone.mask_token = nn.Parameter(torch.ones(1))
        self.backbone.pos_embed = nn.Parameter(torch.ones(1))
        self.backbone.stem_norm = nn.Parameter(torch.ones(1))
        self.backbone.downsample_norm0 = nn.BatchNorm2d(2)
        self.backbone.downsample_norm1 = nn.BatchNorm2d(2)
        self.backbone.downsample_norm2 = nn.BatchNorm2d(2)
        self.backbone.lin = nn.Parameter(torch.ones(1))
        self.backbone.lin.requires_grad = False

        self.backbone.downsample_layers = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=True))
            self.backbone.downsample_layers.append(stage)

        self.decode_head = nn.Conv2d(2, 2, kernel_size=1, groups=2)


class PseudoDataParallel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ConvNeXtExampleModel()

    def forward(self, x):
        return x


def check_convnext_adamw_optimizer(optimizer, gt_lst):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer.param_groups
    assert len(param_groups) == 12
    for i, param_dict in enumerate(param_groups):
        assert param_dict['weight_decay'] == gt_lst[i]['weight_decay']
        assert param_dict['lr_scale'] == gt_lst[i]['lr_scale']
        assert param_dict['lr_scale'] == param_dict['lr']


def test_convnext_learning_rate_decay_optimizer_constructor():

    # paramwise_cfg with ConvNeXtExampleModel
    model = ConvNeXtExampleModel()
    optimizer_cfg = dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05)
    stagewise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='stage_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, stagewise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_convnext_adamw_optimizer(optimizer, stage_wise_gt_lst)

    layerwise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='layer_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optimizer_cfg, layerwise_paramwise_cfg)
    optimizer = optim_constructor(model)
    check_convnext_adamw_optimizer(optimizer, layer_wise_gt_lst)
