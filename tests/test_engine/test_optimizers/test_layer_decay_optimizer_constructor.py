# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.engine import LearningRateDecayOptimizerConstructor

base_lr = 1
decay_rate = 2
base_wd = 0.05
weight_decay = 0.05

expected_stage_wise_lr_wd_convnext = [{
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

expected_layer_wise_lr_wd_convnext = [{
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


class ToyConvNeXt(nn.Module):

    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(ConvModule(3, 4, kernel_size=1, bias=True))
            self.stages.append(stage)
        self.norm0 = nn.BatchNorm2d(2)

        # add some variables to meet unit test coverate rate
        self.cls_token = nn.Parameter(torch.ones(1))
        self.mask_token = nn.Parameter(torch.ones(1))
        self.pos_embed = nn.Parameter(torch.ones(1))
        self.stem_norm = nn.Parameter(torch.ones(1))
        self.downsample_norm0 = nn.BatchNorm2d(2)
        self.downsample_norm1 = nn.BatchNorm2d(2)
        self.downsample_norm2 = nn.BatchNorm2d(2)
        self.lin = nn.Parameter(torch.ones(1))
        self.lin.requires_grad = False
        self.downsample_layers = nn.ModuleList()
        for _ in range(4):
            stage = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=True))
            self.downsample_layers.append(stage)


class ToyDetector(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv2d(2, 2, kernel_size=1, groups=2)


class PseudoDataParallel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.module = model


def check_optimizer_lr_wd(optimizer, gt_lr_wd):
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == base_lr
    assert optimizer.defaults['weight_decay'] == base_wd
    param_groups = optimizer.param_groups
    print(param_groups)
    assert len(param_groups) == len(gt_lr_wd)
    for i, param_dict in enumerate(param_groups):
        assert param_dict['weight_decay'] == gt_lr_wd[i]['weight_decay']
        assert param_dict['lr_scale'] == gt_lr_wd[i]['lr_scale']
        assert param_dict['lr_scale'] == param_dict['lr']


def test_learning_rate_decay_optimizer_constructor():

    # Test lr wd for ConvNeXT
    backbone = ToyConvNeXt()
    model = PseudoDataParallel(ToyDetector(backbone))
    optim_wrapper_cfg = dict(
        type='OptimWrapper',
        optimizer=dict(
            type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05))
    # stagewise decay
    stagewise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='stage_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optim_wrapper_cfg, stagewise_paramwise_cfg)
    optim_wrapper = optim_constructor(model)
    check_optimizer_lr_wd(optim_wrapper.optimizer,
                          expected_stage_wise_lr_wd_convnext)
    # layerwise decay
    layerwise_paramwise_cfg = dict(
        decay_rate=decay_rate, decay_type='layer_wise', num_layers=6)
    optim_constructor = LearningRateDecayOptimizerConstructor(
        optim_wrapper_cfg, layerwise_paramwise_cfg)
    optim_wrapper = optim_constructor(model)
    check_optimizer_lr_wd(optim_wrapper.optimizer,
                          expected_layer_wise_lr_wd_convnext)
