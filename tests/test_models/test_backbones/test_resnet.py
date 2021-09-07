# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv import assert_params_all_zeros
from mmcv.ops import DeformConv2dPack
from torch.nn.modules import AvgPool2d, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones import ResNet, ResNetV1d
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock
from .utils import check_norm_state, is_block, is_norm


def test_resnet_basic_block():
    with pytest.raises(AssertionError):
        # Not implemented yet.
        dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
        BasicBlock(64, 64, dcn=dcn)

    with pytest.raises(AssertionError):
        # Not implemented yet.
        plugins = [
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv3')
        ]
        BasicBlock(64, 64, plugins=plugins)

    with pytest.raises(AssertionError):
        # Not implemented yet
        plugins = [
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0010',
                    kv_stride=2),
                position='after_conv2')
        ]
        BasicBlock(64, 64, plugins=plugins)

    # test BasicBlock structure and forward
    block = BasicBlock(64, 64)
    assert block.conv1.in_channels == 64
    assert block.conv1.out_channels == 64
    assert block.conv1.kernel_size == (3, 3)
    assert block.conv2.in_channels == 64
    assert block.conv2.out_channels == 64
    assert block.conv2.kernel_size == (3, 3)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test BasicBlock with checkpoint forward
    block = BasicBlock(64, 64, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_resnet_bottleneck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        Bottleneck(64, 64, style='tensorflow')

    with pytest.raises(AssertionError):
        # Allowed positions are 'after_conv1', 'after_conv2', 'after_conv3'
        plugins = [
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv4')
        ]
        Bottleneck(64, 16, plugins=plugins)

    with pytest.raises(AssertionError):
        # Need to specify different postfix to avoid duplicate plugin name
        plugins = [
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv3'),
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv3')
        ]
        Bottleneck(64, 16, plugins=plugins)

    with pytest.raises(KeyError):
        # Plugin type is not supported
        plugins = [dict(cfg=dict(type='WrongPlugin'), position='after_conv3')]
        Bottleneck(64, 16, plugins=plugins)

    # Test Bottleneck with checkpoint forward
    block = Bottleneck(64, 16, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test Bottleneck style
    block = Bottleneck(64, 64, stride=2, style='pytorch')
    assert block.conv1.stride == (1, 1)
    assert block.conv2.stride == (2, 2)
    block = Bottleneck(64, 64, stride=2, style='caffe')
    assert block.conv1.stride == (2, 2)
    assert block.conv2.stride == (1, 1)

    # Test Bottleneck DCN
    dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        Bottleneck(64, 64, dcn=dcn, conv_cfg=dict(type='Conv'))
    block = Bottleneck(64, 64, dcn=dcn)
    assert isinstance(block.conv2, DeformConv2dPack)

    # Test Bottleneck forward
    block = Bottleneck(64, 16)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test Bottleneck with 1 ContextBlock after conv3
    plugins = [
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16),
            position='after_conv3')
    ]
    block = Bottleneck(64, 16, plugins=plugins)
    assert block.context_block.in_channels == 64
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test Bottleneck with 1 GeneralizedAttention after conv2
    plugins = [
        dict(
            cfg=dict(
                type='GeneralizedAttention',
                spatial_range=-1,
                num_heads=8,
                attention_type='0010',
                kv_stride=2),
            position='after_conv2')
    ]
    block = Bottleneck(64, 16, plugins=plugins)
    assert block.gen_attention_block.in_channels == 16
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test Bottleneck with 1 GeneralizedAttention after conv2, 1 NonLocal2D
    # after conv2, 1 ContextBlock after conv3
    plugins = [
        dict(
            cfg=dict(
                type='GeneralizedAttention',
                spatial_range=-1,
                num_heads=8,
                attention_type='0010',
                kv_stride=2),
            position='after_conv2'),
        dict(cfg=dict(type='NonLocal2d'), position='after_conv2'),
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16),
            position='after_conv3')
    ]
    block = Bottleneck(64, 16, plugins=plugins)
    assert block.gen_attention_block.in_channels == 16
    assert block.nonlocal_block.in_channels == 16
    assert block.context_block.in_channels == 64
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test Bottleneck with 1 ContextBlock after conv2, 2 ContextBlock after
    # conv3
    plugins = [
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16, postfix=1),
            position='after_conv2'),
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16, postfix=2),
            position='after_conv3'),
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16, postfix=3),
            position='after_conv3')
    ]
    block = Bottleneck(64, 16, plugins=plugins)
    assert block.context_block1.in_channels == 16
    assert block.context_block2.in_channels == 64
    assert block.context_block3.in_channels == 64
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_simplied_basic_block():
    with pytest.raises(AssertionError):
        # Not implemented yet.
        dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
        SimplifiedBasicBlock(64, 64, dcn=dcn)

    with pytest.raises(AssertionError):
        # Not implemented yet.
        plugins = [
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv3')
        ]
        SimplifiedBasicBlock(64, 64, plugins=plugins)

    with pytest.raises(AssertionError):
        # Not implemented yet
        plugins = [
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0010',
                    kv_stride=2),
                position='after_conv2')
        ]
        SimplifiedBasicBlock(64, 64, plugins=plugins)

    with pytest.raises(AssertionError):
        # Not implemented yet
        SimplifiedBasicBlock(64, 64, with_cp=True)

    # test SimplifiedBasicBlock structure and forward
    block = SimplifiedBasicBlock(64, 64)
    assert block.conv1.in_channels == 64
    assert block.conv1.out_channels == 64
    assert block.conv1.kernel_size == (3, 3)
    assert block.conv2.in_channels == 64
    assert block.conv2.out_channels == 64
    assert block.conv2.kernel_size == (3, 3)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # test SimplifiedBasicBlock without norm
    block = SimplifiedBasicBlock(64, 64, norm_cfg=None)
    assert block.norm1 is None
    assert block.norm2 is None
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_resnet_res_layer():
    # Test ResLayer of 3 Bottleneck w\o downsample
    layer = ResLayer(Bottleneck, 64, 16, 3)
    assert len(layer) == 3
    assert layer[0].conv1.in_channels == 64
    assert layer[0].conv1.out_channels == 16
    for i in range(1, len(layer)):
        assert layer[i].conv1.in_channels == 64
        assert layer[i].conv1.out_channels == 16
    for i in range(len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test ResLayer of 3 Bottleneck with downsample
    layer = ResLayer(Bottleneck, 64, 64, 3)
    assert layer[0].downsample[0].out_channels == 256
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 56, 56])

    # Test ResLayer of 3 Bottleneck with stride=2
    layer = ResLayer(Bottleneck, 64, 64, 3, stride=2)
    assert layer[0].downsample[0].out_channels == 256
    assert layer[0].downsample[0].stride == (2, 2)
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 28, 28])

    # Test ResLayer of 3 Bottleneck with stride=2 and average downsample
    layer = ResLayer(Bottleneck, 64, 64, 3, stride=2, avg_down=True)
    assert isinstance(layer[0].downsample[0], AvgPool2d)
    assert layer[0].downsample[1].out_channels == 256
    assert layer[0].downsample[1].stride == (1, 1)
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 28, 28])

    # Test ResLayer of 3 BasicBlock with stride=2 and downsample_first=False
    layer = ResLayer(BasicBlock, 64, 64, 3, stride=2, downsample_first=False)
    assert layer[2].downsample[0].out_channels == 64
    assert layer[2].downsample[0].stride == (2, 2)
    for i in range(len(layer) - 1):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 64, 28, 28])


def test_resnest_stem():
    # Test default stem_channels
    model = ResNet(50)
    assert model.stem_channels == 64
    assert model.conv1.out_channels == 64
    assert model.norm1.num_features == 64

    # Test default stem_channels, with base_channels=3
    model = ResNet(50, base_channels=3)
    assert model.stem_channels == 3
    assert model.conv1.out_channels == 3
    assert model.norm1.num_features == 3
    assert model.layer1[0].conv1.in_channels == 3

    # Test stem_channels=3
    model = ResNet(50, stem_channels=3)
    assert model.stem_channels == 3
    assert model.conv1.out_channels == 3
    assert model.norm1.num_features == 3
    assert model.layer1[0].conv1.in_channels == 3

    # Test stem_channels=3, with base_channels=2
    model = ResNet(50, stem_channels=3, base_channels=2)
    assert model.stem_channels == 3
    assert model.conv1.out_channels == 3
    assert model.norm1.num_features == 3
    assert model.layer1[0].conv1.in_channels == 3

    # Test V1d stem_channels
    model = ResNetV1d(depth=50, stem_channels=6)
    model.train()
    assert model.stem[0].out_channels == 3
    assert model.stem[1].num_features == 3
    assert model.stem[3].out_channels == 3
    assert model.stem[4].num_features == 3
    assert model.stem[6].out_channels == 6
    assert model.stem[7].num_features == 6
    assert model.layer1[0].conv1.in_channels == 6


def test_resnet_backbone():
    """Test resnet backbone."""
    with pytest.raises(KeyError):
        # ResNet depth should be in [18, 34, 50, 101, 152]
        ResNet(20)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=0)

    with pytest.raises(AssertionError):
        # len(stage_with_dcn) == num_stages
        dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
        ResNet(50, dcn=dcn, stage_with_dcn=(True, ))

    with pytest.raises(AssertionError):
        # len(stage_with_plugin) == num_stages
        plugins = [
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                stages=(False, True, True),
                position='after_conv3')
        ]
        ResNet(50, plugins=plugins)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=5)

    with pytest.raises(AssertionError):
        # len(strides) == len(dilations) == num_stages
        ResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = ResNet(50, pretrained=0)

    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        ResNet(50, style='tensorflow')

    # Test ResNet50 norm_eval=True
    model = ResNet(50, norm_eval=True, base_channels=1)
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ResNet50 with torchvision pretrained weight
    model = ResNet(
        depth=50, norm_eval=True, pretrained='torchvision://resnet50')
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ResNet50 with first stage frozen
    frozen_stages = 1
    model = ResNet(50, frozen_stages=frozen_stages, base_channels=1)
    model.train()
    assert model.norm1.training is False
    for layer in [model.conv1, model.norm1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test ResNet50V1d with first stage frozen
    model = ResNetV1d(depth=50, frozen_stages=frozen_stages, base_channels=2)
    assert len(model.stem) == 9
    model.train()
    assert check_norm_state(model.stem, False)
    for param in model.stem.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test ResNet18 forward
    model = ResNet(18)
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 64, 8, 8])
    assert feat[1].shape == torch.Size([1, 128, 4, 4])
    assert feat[2].shape == torch.Size([1, 256, 2, 2])
    assert feat[3].shape == torch.Size([1, 512, 1, 1])

    # Test ResNet18 with checkpoint forward
    model = ResNet(18, with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp

    # Test ResNet50 with BatchNorm forward
    model = ResNet(50, base_channels=1)
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 4, 8, 8])
    assert feat[1].shape == torch.Size([1, 8, 4, 4])
    assert feat[2].shape == torch.Size([1, 16, 2, 2])
    assert feat[3].shape == torch.Size([1, 32, 1, 1])

    # Test ResNet50 with layers 1, 2, 3 out forward
    model = ResNet(50, out_indices=(0, 1, 2), base_channels=1)
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 4, 8, 8])
    assert feat[1].shape == torch.Size([1, 8, 4, 4])
    assert feat[2].shape == torch.Size([1, 16, 2, 2])

    # Test ResNet50 with checkpoint forward
    model = ResNet(50, with_cp=True, base_channels=1)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 4, 8, 8])
    assert feat[1].shape == torch.Size([1, 8, 4, 4])
    assert feat[2].shape == torch.Size([1, 16, 2, 2])
    assert feat[3].shape == torch.Size([1, 32, 1, 1])

    # Test ResNet50 with GroupNorm forward
    model = ResNet(
        50,
        base_channels=4,
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 16, 8, 8])
    assert feat[1].shape == torch.Size([1, 32, 4, 4])
    assert feat[2].shape == torch.Size([1, 64, 2, 2])
    assert feat[3].shape == torch.Size([1, 128, 1, 1])

    # Test ResNet50 with 1 GeneralizedAttention after conv2, 1 NonLocal2D
    # after conv2, 1 ContextBlock after conv3 in layers 2, 3, 4
    plugins = [
        dict(
            cfg=dict(
                type='GeneralizedAttention',
                spatial_range=-1,
                num_heads=8,
                attention_type='0010',
                kv_stride=2),
            stages=(False, True, True, True),
            position='after_conv2'),
        dict(cfg=dict(type='NonLocal2d'), position='after_conv2'),
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16),
            stages=(False, True, True, False),
            position='after_conv3')
    ]
    model = ResNet(50, plugins=plugins, base_channels=8)
    for m in model.layer1.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert not hasattr(m, 'gen_attention_block')
            assert m.nonlocal_block.in_channels == 8
    for m in model.layer2.modules():
        if is_block(m):
            assert m.nonlocal_block.in_channels == 16
            assert m.gen_attention_block.in_channels == 16
            assert m.context_block.in_channels == 64

    for m in model.layer3.modules():
        if is_block(m):
            assert m.nonlocal_block.in_channels == 32
            assert m.gen_attention_block.in_channels == 32
            assert m.context_block.in_channels == 128

    for m in model.layer4.modules():
        if is_block(m):
            assert m.nonlocal_block.in_channels == 64
            assert m.gen_attention_block.in_channels == 64
            assert not hasattr(m, 'context_block')
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 32, 8, 8])
    assert feat[1].shape == torch.Size([1, 64, 4, 4])
    assert feat[2].shape == torch.Size([1, 128, 2, 2])
    assert feat[3].shape == torch.Size([1, 256, 1, 1])

    # Test ResNet50 with 1 ContextBlock after conv2, 1 ContextBlock after
    # conv3 in layers 2, 3, 4
    plugins = [
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16, postfix=1),
            stages=(False, True, True, False),
            position='after_conv3'),
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16, postfix=2),
            stages=(False, True, True, False),
            position='after_conv3')
    ]

    model = ResNet(50, plugins=plugins, base_channels=8)
    for m in model.layer1.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert not hasattr(m, 'context_block1')
            assert not hasattr(m, 'context_block2')
    for m in model.layer2.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert m.context_block1.in_channels == 64
            assert m.context_block2.in_channels == 64

    for m in model.layer3.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert m.context_block1.in_channels == 128
            assert m.context_block2.in_channels == 128

    for m in model.layer4.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert not hasattr(m, 'context_block1')
            assert not hasattr(m, 'context_block2')
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 32, 8, 8])
    assert feat[1].shape == torch.Size([1, 64, 4, 4])
    assert feat[2].shape == torch.Size([1, 128, 2, 2])
    assert feat[3].shape == torch.Size([1, 256, 1, 1])

    # Test ResNet50 zero initialization of residual
    model = ResNet(50, zero_init_residual=True, base_channels=1)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            assert assert_params_all_zeros(m.norm3)
        elif isinstance(m, BasicBlock):
            assert assert_params_all_zeros(m.norm2)
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 4, 8, 8])
    assert feat[1].shape == torch.Size([1, 8, 4, 4])
    assert feat[2].shape == torch.Size([1, 16, 2, 2])
    assert feat[3].shape == torch.Size([1, 32, 1, 1])

    # Test ResNetV1d forward
    model = ResNetV1d(depth=50, base_channels=2)
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 8, 8, 8])
    assert feat[1].shape == torch.Size([1, 16, 4, 4])
    assert feat[2].shape == torch.Size([1, 32, 2, 2])
    assert feat[3].shape == torch.Size([1, 64, 1, 1])
