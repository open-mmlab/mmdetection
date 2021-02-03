import pytest
import torch
from mmcv.ops import DeformConv2dPack
from torch.nn.modules import AvgPool2d, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones import (RegNet, Res2Net, ResNeSt, ResNet,
                                    ResNetV1d, ResNeXt, TridentResNet)
from mmdet.models.backbones.hourglass import HourglassNet
from mmdet.models.backbones.res2net import Bottle2neck
from mmdet.models.backbones.resnest import Bottleneck as BottleneckS
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.models.backbones.resnext import Bottleneck as BottleneckX
from mmdet.models.backbones.trident_resnet import TridentBottleneck
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock


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


def all_zeros(modules):
    """Check if the weight(and bias) is all zero."""
    weight_zero = torch.allclose(modules.weight.data,
                                 torch.zeros_like(modules.weight.data))
    if hasattr(modules, 'bias'):
        bias_zero = torch.allclose(modules.bias.data,
                                   torch.zeros_like(modules.bias.data))
    else:
        bias_zero = True

    return weight_zero and bias_zero


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


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


def test_trident_resnet_bottleneck():
    trident_dilations = (1, 2, 3)
    test_branch_idx = 1
    concat_output = True
    trident_build_config = (trident_dilations, test_branch_idx, concat_output)

    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        TridentBottleneck(
            *trident_build_config, inplanes=64, planes=64, style='tensorflow')

    with pytest.raises(AssertionError):
        # Allowed positions are 'after_conv1', 'after_conv2', 'after_conv3'
        plugins = [
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                position='after_conv4')
        ]
        TridentBottleneck(
            *trident_build_config, inplanes=64, planes=16, plugins=plugins)

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
        TridentBottleneck(
            *trident_build_config, inplanes=64, planes=16, plugins=plugins)

    with pytest.raises(KeyError):
        # Plugin type is not supported
        plugins = [dict(cfg=dict(type='WrongPlugin'), position='after_conv3')]
        TridentBottleneck(
            *trident_build_config, inplanes=64, planes=16, plugins=plugins)

    # Test Bottleneck with checkpoint forward
    block = TridentBottleneck(
        *trident_build_config, inplanes=64, planes=16, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([block.num_branch, 64, 56, 56])

    # Test Bottleneck style
    block = TridentBottleneck(
        *trident_build_config,
        inplanes=64,
        planes=64,
        stride=2,
        style='pytorch')
    assert block.conv1.stride == (1, 1)
    assert block.conv2.stride == (2, 2)
    block = TridentBottleneck(
        *trident_build_config, inplanes=64, planes=64, stride=2, style='caffe')
    assert block.conv1.stride == (2, 2)
    assert block.conv2.stride == (1, 1)

    # Test Bottleneck forward
    block = TridentBottleneck(*trident_build_config, inplanes=64, planes=16)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([block.num_branch, 64, 56, 56])

    # Test Bottleneck with 1 ContextBlock after conv3
    plugins = [
        dict(
            cfg=dict(type='ContextBlock', ratio=1. / 16),
            position='after_conv3')
    ]
    block = TridentBottleneck(
        *trident_build_config, inplanes=64, planes=16, plugins=plugins)
    assert block.context_block.in_channels == 64
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([block.num_branch, 64, 56, 56])

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
    block = TridentBottleneck(
        *trident_build_config, inplanes=64, planes=16, plugins=plugins)
    assert block.gen_attention_block.in_channels == 16
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([block.num_branch, 64, 56, 56])

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
    block = TridentBottleneck(
        *trident_build_config, inplanes=64, planes=16, plugins=plugins)
    assert block.gen_attention_block.in_channels == 16
    assert block.nonlocal_block.in_channels == 16
    assert block.context_block.in_channels == 64
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([block.num_branch, 64, 56, 56])

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
    block = TridentBottleneck(
        *trident_build_config, inplanes=64, planes=16, plugins=plugins)
    assert block.context_block1.in_channels == 16
    assert block.context_block2.in_channels == 64
    assert block.context_block3.in_channels == 64
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([block.num_branch, 64, 56, 56])


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


def test_trident_resnet_backbone():
    tridentresnet_config = dict(
        num_branch=3,
        test_branch_idx=1,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        trident_dilations=(1, 2, 3),
        out_indices=(2, ),
    )
    """Test tridentresnet backbone."""
    with pytest.raises(AssertionError):
        # TridentResNet depth should be in [50, 101, 152]
        TridentResNet(18, **tridentresnet_config)

    with pytest.raises(AssertionError):
        # In TridentResNet: num_stages == 3
        TridentResNet(50, num_stages=4, **tridentresnet_config)

    model = TridentResNet(50, num_stages=3, **tridentresnet_config)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([3, 1024, 14, 14])


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

    # Test default stem_channels, with base_channels=32
    model = ResNet(50, base_channels=32)
    assert model.stem_channels == 32
    assert model.conv1.out_channels == 32
    assert model.norm1.num_features == 32
    assert model.layer1[0].conv1.in_channels == 32

    # Test stem_channels=64
    model = ResNet(50, stem_channels=64)
    assert model.stem_channels == 64
    assert model.conv1.out_channels == 64
    assert model.norm1.num_features == 64
    assert model.layer1[0].conv1.in_channels == 64

    # Test stem_channels=64, with base_channels=32
    model = ResNet(50, stem_channels=64, base_channels=32)
    assert model.stem_channels == 64
    assert model.conv1.out_channels == 64
    assert model.norm1.num_features == 64
    assert model.layer1[0].conv1.in_channels == 64

    # Test stem_channels=128
    model = ResNet(depth=50, stem_channels=128)
    model.init_weights()
    model.train()
    assert model.conv1.out_channels == 128
    assert model.layer1[0].conv1.in_channels == 128

    # Test V1d stem_channels
    model = ResNetV1d(depth=50, stem_channels=128)
    model.init_weights()
    model.train()
    assert model.stem[0].out_channels == 64
    assert model.stem[1].num_features == 64
    assert model.stem[3].out_channels == 64
    assert model.stem[4].num_features == 64
    assert model.stem[6].out_channels == 128
    assert model.stem[7].num_features == 128
    assert model.layer1[0].conv1.in_channels == 128


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
        model = ResNet(50)
        model.init_weights(pretrained=0)

    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        ResNet(50, style='tensorflow')

    # Test ResNet50 norm_eval=True
    model = ResNet(50, norm_eval=True)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ResNet50 with torchvision pretrained weight
    model = ResNet(depth=50, norm_eval=True)
    model.init_weights('torchvision://resnet50')
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ResNet50 with first stage frozen
    frozen_stages = 1
    model = ResNet(50, frozen_stages=frozen_stages)
    model.init_weights()
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
    model = ResNetV1d(depth=50, frozen_stages=frozen_stages)
    assert len(model.stem) == 9
    model.init_weights()
    model.train()
    check_norm_state(model.stem, False)
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
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 64, 56, 56])
    assert feat[1].shape == torch.Size([1, 128, 28, 28])
    assert feat[2].shape == torch.Size([1, 256, 14, 14])
    assert feat[3].shape == torch.Size([1, 512, 7, 7])

    # Test ResNet18 with checkpoint forward
    model = ResNet(18, with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp

    # Test ResNet50 with BatchNorm forward
    model = ResNet(50)
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    # Test ResNet50 with layers 1, 2, 3 out forward
    model = ResNet(50, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])

    # Test ResNet50 with checkpoint forward
    model = ResNet(50, with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    # Test ResNet50 with GroupNorm forward
    model = ResNet(
        50, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

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
    model = ResNet(50, plugins=plugins)
    for m in model.layer1.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert not hasattr(m, 'gen_attention_block')
            assert m.nonlocal_block.in_channels == 64
    for m in model.layer2.modules():
        if is_block(m):
            assert m.nonlocal_block.in_channels == 128
            assert m.gen_attention_block.in_channels == 128
            assert m.context_block.in_channels == 512

    for m in model.layer3.modules():
        if is_block(m):
            assert m.nonlocal_block.in_channels == 256
            assert m.gen_attention_block.in_channels == 256
            assert m.context_block.in_channels == 1024

    for m in model.layer4.modules():
        if is_block(m):
            assert m.nonlocal_block.in_channels == 512
            assert m.gen_attention_block.in_channels == 512
            assert not hasattr(m, 'context_block')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

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

    model = ResNet(50, plugins=plugins)
    for m in model.layer1.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert not hasattr(m, 'context_block1')
            assert not hasattr(m, 'context_block2')
    for m in model.layer2.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert m.context_block1.in_channels == 512
            assert m.context_block2.in_channels == 512

    for m in model.layer3.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert m.context_block1.in_channels == 1024
            assert m.context_block2.in_channels == 1024

    for m in model.layer4.modules():
        if is_block(m):
            assert not hasattr(m, 'context_block')
            assert not hasattr(m, 'context_block1')
            assert not hasattr(m, 'context_block2')
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    # Test ResNet50 zero initialization of residual
    model = ResNet(50, zero_init_residual=True)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            assert all_zeros(m.norm3)
        elif isinstance(m, BasicBlock):
            assert all_zeros(m.norm2)
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    # Test ResNetV1d forward
    model = ResNetV1d(depth=50)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])


def test_renext_bottleneck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        BottleneckX(64, 64, groups=32, base_width=4, style='tensorflow')

    # Test ResNeXt Bottleneck structure
    block = BottleneckX(
        64, 64, groups=32, base_width=4, stride=2, style='pytorch')
    assert block.conv2.stride == (2, 2)
    assert block.conv2.groups == 32
    assert block.conv2.out_channels == 128

    # Test ResNeXt Bottleneck with DCN
    dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        # conv_cfg must be None if dcn is not None
        BottleneckX(
            64,
            64,
            groups=32,
            base_width=4,
            dcn=dcn,
            conv_cfg=dict(type='Conv'))
    BottleneckX(64, 64, dcn=dcn)

    # Test ResNeXt Bottleneck forward
    block = BottleneckX(64, 16, groups=32, base_width=4)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    # Test ResNeXt Bottleneck forward with plugins
    plugins = [
        dict(
            cfg=dict(
                type='GeneralizedAttention',
                spatial_range=-1,
                num_heads=8,
                attention_type='0010',
                kv_stride=2),
            stages=(False, False, True, True),
            position='after_conv2')
    ]
    block = BottleneckX(64, 16, groups=32, base_width=4, plugins=plugins)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_resnext_backbone():
    with pytest.raises(KeyError):
        # ResNeXt depth should be in [50, 101, 152]
        ResNeXt(depth=18)

    # Test ResNeXt with group 32, base_width 4
    model = ResNeXt(depth=50, groups=32, base_width=4)
    for m in model.modules():
        if is_block(m):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])


regnet_test_data = [
    ('regnetx_400mf',
     dict(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22,
          bot_mul=1.0), [32, 64, 160, 384]),
    ('regnetx_800mf',
     dict(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16,
          bot_mul=1.0), [64, 128, 288, 672]),
    ('regnetx_1.6gf',
     dict(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18,
          bot_mul=1.0), [72, 168, 408, 912]),
    ('regnetx_3.2gf',
     dict(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25,
          bot_mul=1.0), [96, 192, 432, 1008]),
    ('regnetx_4.0gf',
     dict(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23,
          bot_mul=1.0), [80, 240, 560, 1360]),
    ('regnetx_6.4gf',
     dict(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17,
          bot_mul=1.0), [168, 392, 784, 1624]),
    ('regnetx_8.0gf',
     dict(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23,
          bot_mul=1.0), [80, 240, 720, 1920]),
    ('regnetx_12gf',
     dict(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19,
          bot_mul=1.0), [224, 448, 896, 2240]),
]


@pytest.mark.parametrize('arch_name,arch,out_channels', regnet_test_data)
def test_regnet_backbone(arch_name, arch, out_channels):
    with pytest.raises(AssertionError):
        # ResNeXt depth should be in [50, 101, 152]
        RegNet(arch_name + '233')

    # Test RegNet with arch_name
    model = RegNet(arch_name)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, out_channels[0], 56, 56])
    assert feat[1].shape == torch.Size([1, out_channels[1], 28, 28])
    assert feat[2].shape == torch.Size([1, out_channels[2], 14, 14])
    assert feat[3].shape == torch.Size([1, out_channels[3], 7, 7])

    # Test RegNet with arch
    model = RegNet(arch)
    assert feat[0].shape == torch.Size([1, out_channels[0], 56, 56])
    assert feat[1].shape == torch.Size([1, out_channels[1], 28, 28])
    assert feat[2].shape == torch.Size([1, out_channels[2], 14, 14])
    assert feat[3].shape == torch.Size([1, out_channels[3], 7, 7])


def test_res2net_bottle2neck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        Bottle2neck(64, 64, base_width=26, scales=4, style='tensorflow')

    with pytest.raises(AssertionError):
        # Scale must be larger than 1
        Bottle2neck(64, 64, base_width=26, scales=1, style='pytorch')

    # Test Res2Net Bottle2neck structure
    block = Bottle2neck(
        64, 64, base_width=26, stride=2, scales=4, style='pytorch')
    assert block.scales == 4

    # Test Res2Net Bottle2neck with DCN
    dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        # conv_cfg must be None if dcn is not None
        Bottle2neck(
            64,
            64,
            base_width=26,
            scales=4,
            dcn=dcn,
            conv_cfg=dict(type='Conv'))
    Bottle2neck(64, 64, dcn=dcn)

    # Test Res2Net Bottle2neck forward
    block = Bottle2neck(64, 16, base_width=26, scales=4)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_res2net_backbone():
    with pytest.raises(KeyError):
        # Res2Net depth should be in [50, 101, 152]
        Res2Net(depth=18)

    # Test Res2Net with scales 4, base_width 26
    model = Res2Net(depth=50, scales=4, base_width=26)
    for m in model.modules():
        if is_block(m):
            assert m.scales == 4
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])
    assert feat[3].shape == torch.Size([1, 2048, 7, 7])


def test_hourglass_backbone():
    with pytest.raises(AssertionError):
        # HourglassNet's num_stacks should larger than 0
        HourglassNet(num_stacks=0)

    with pytest.raises(AssertionError):
        # len(stage_channels) should equal len(stage_blocks)
        HourglassNet(
            stage_channels=[256, 256, 384, 384, 384],
            stage_blocks=[2, 2, 2, 2, 2, 4])

    with pytest.raises(AssertionError):
        # len(stage_channels) should lagrer than downsample_times
        HourglassNet(
            downsample_times=5,
            stage_channels=[256, 256, 384, 384, 384],
            stage_blocks=[2, 2, 2, 2, 2])

    # Test HourglassNet-52
    model = HourglassNet(num_stacks=1)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 511, 511)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 256, 128, 128])

    # Test HourglassNet-104
    model = HourglassNet(num_stacks=2)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 511, 511)
    feat = model(imgs)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size([1, 256, 128, 128])
    assert feat[1].shape == torch.Size([1, 256, 128, 128])


def test_resnest_bottleneck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        BottleneckS(64, 64, radix=2, reduction_factor=4, style='tensorflow')

    # Test ResNeSt Bottleneck structure
    block = BottleneckS(
        64, 256, radix=2, reduction_factor=4, stride=2, style='pytorch')
    assert block.avd_layer.stride == 2
    assert block.conv2.channels == 256

    # Test ResNeSt Bottleneck forward
    block = BottleneckS(64, 16, radix=2, reduction_factor=4)
    x = torch.randn(2, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 64, 56, 56])


def test_resnest_backbone():
    with pytest.raises(KeyError):
        # ResNeSt depth should be in [50, 101, 152, 200]
        ResNeSt(depth=18)

    # Test ResNeSt with radix 2, reduction_factor 4
    model = ResNeSt(
        depth=50, radix=2, reduction_factor=4, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([2, 256, 56, 56])
    assert feat[1].shape == torch.Size([2, 512, 28, 28])
    assert feat[2].shape == torch.Size([2, 1024, 14, 14])
    assert feat[3].shape == torch.Size([2, 2048, 7, 7])
