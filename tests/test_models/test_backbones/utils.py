# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones.res2net import Bottle2neck
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.models.backbones.resnext import Bottleneck as BottleneckX
from mmdet.models.utils import SimplifiedBasicBlock


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (BasicBlock, Bottleneck, BottleneckX, Bottle2neck,
                            SimplifiedBasicBlock)):
        return True
    return False


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True
