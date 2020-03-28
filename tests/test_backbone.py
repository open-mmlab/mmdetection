import pytest
import torch
from torch.nn.modules import AvgPool2d, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones import ResNet, ResNetV1d, ResNeXt
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.models.backbones.resnext import Bottleneck as BottleneckX
from mmdet.models.utils import ResLayer
from mmdet.ops import DeformConvPack


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (BasicBlock, Bottleneck, BottleneckX)):
        return True
    return False


def is_norm(modules):
    """Check if is one of the norms."""
    if isinstance(modules, (GroupNorm, _BatchNorm)):
        return True
    return False


def all_zeros(modules):
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
        BasicBlock(64, 64, with_cp=True)

    with pytest.raises(AssertionError):
        # Not implemented yet.
        dcn = dict(type='DCN', deformable_groups=1, fallback_on_stride=False)
        BasicBlock(64, 64, dcn=dcn)

    with pytest.raises(AssertionError):
        # Not implemented yet.
        gcb = dict(ratio=1. / 4., )
        BasicBlock(64, 64, gcb=gcb)

    with pytest.raises(AssertionError):
        # Not implemented yet
        gen_attention = dict(
            spatial_range=-1, num_heads=8, attention_type='0010', kv_stride=2)
        BasicBlock(64, 64, gen_attention=gen_attention)

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


def test_resnet_bottleneck():

    with pytest.raises(AssertionError):
        # style must be in ['pytorch', 'caffe']
        Bottleneck(64, 64, style='tensorflow')

    block = Bottleneck(64, 16, with_cp=True)
    assert block.with_cp
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    block = Bottleneck(64, 64, stride=2, style='pytorch')
    assert block.conv1.stride == (1, 1)
    assert block.conv2.stride == (2, 2)
    block = Bottleneck(64, 64, stride=2, style='caffe')
    assert block.conv1.stride == (2, 2)
    assert block.conv2.stride == (1, 1)

    dcn = dict(type='DCN', deformable_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        Bottleneck(64, 64, dcn=dcn, conv_cfg=dict(type='Conv'))
    block = Bottleneck(64, 64, dcn=dcn)
    assert isinstance(block.conv2, DeformConvPack)

    block = Bottleneck(64, 16)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    gcb = dict(ratio=1. / 4., )
    block = Bottleneck(64, 16, gcb=gcb)
    assert hasattr(block, 'context_block')
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])

    gen_attention = dict(
        spatial_range=-1, num_heads=8, attention_type='0010', kv_stride=2)
    block = Bottleneck(64, 16, gen_attention=gen_attention)
    assert hasattr(block, 'gen_attention')
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_resnet_res_layer():
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

    layer = ResLayer(Bottleneck, 64, 64, 3)
    assert layer[0].downsample[0].out_channels == 256
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 56, 56])

    layer = ResLayer(Bottleneck, 64, 64, 3, stride=2)
    assert layer[0].downsample[0].out_channels == 256
    assert layer[0].downsample[0].stride == (2, 2)
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 28, 28])

    layer = ResLayer(Bottleneck, 64, 64, 3, stride=2, avg_down=True)
    assert isinstance(layer[0].downsample[0], AvgPool2d)
    assert layer[0].downsample[1].out_channels == 256
    assert layer[0].downsample[1].stride == (1, 1)
    for i in range(1, len(layer)):
        assert layer[i].downsample is None
    x = torch.randn(1, 64, 56, 56)
    x_out = layer(x)
    assert x_out.shape == torch.Size([1, 256, 28, 28])


def test_resnet_backbone():
    """Test resnet backbone"""
    with pytest.raises(KeyError):
        # ResNet depth should be in [18, 34, 50, 101, 152]
        ResNet(20)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=0)

    with pytest.raises(AssertionError):
        ResNet(18, with_cp=True)

    with pytest.raises(AssertionError):
        # len(stage_with_dcn) == num_stages
        dcn = dict(type='DCN', deformable_groups=1, fallback_on_stride=False)
        ResNet(50, dcn=dcn, stage_with_dcn=(True, ))

    with pytest.raises(AssertionError):
        # len(stage_with_gcb) == num_stages
        gcb = dict(ratio=1. / 4., )
        ResNet(50, gcb=gcb, stage_with_gcb=(True, ))

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=5)

    with pytest.raises(AssertionError):
        # len(strides) == len(dilations) == num_stages
        ResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

    with pytest.raises(TypeError):
        model = ResNet(50)
        model.init_weights(pretrained=0)

    with pytest.raises(AssertionError):
        # style must be in ['pytorch', 'caffe']
        ResNet(50, style='tensorflow')

    with pytest.raises(AssertionError):
        # assert not with_cp
        ResNet(18, with_cp=True)

    model = ResNet(18)
    model.init_weights()

    model = ResNet(50, norm_eval=True)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    model = ResNet(depth=50, norm_eval=True)
    model.init_weights('torchvision://resnet50')
    model.train()
    assert check_norm_state(model.modules(), False)

    frozen_stages = 1
    model = ResNet(50, frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    assert model.norm1.training is False
    for layer in [model.conv1, model.norm1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, 'layer{}'.format(i))
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    model = ResNetV1d(depth=50, frozen_stages=frozen_stages)
    assert len(model.stem) == 9
    model.init_weights()
    model.train()
    check_norm_state(model.stem, False)
    for param in model.stem.parameters():
        assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, 'layer{}'.format(i))
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

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

    model = ResNet(50, out_indices=(0, 1, 2))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 256, 56, 56])
    assert feat[1].shape == torch.Size([1, 512, 28, 28])
    assert feat[2].shape == torch.Size([1, 1024, 14, 14])

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


def test_renext_bottleneck():
    with pytest.raises(AssertionError):
        # style must be in ['pytorch', 'caffe']
        BottleneckX(64, 64, groups=32, base_width=4, style='tensorflow')

    block = BottleneckX(
        64, 64, groups=32, base_width=4, stride=2, style='pytorch')
    assert block.conv2.stride == (2, 2)
    assert block.conv2.groups == 32
    assert block.conv2.out_channels == 128

    dcn = dict(type='DCN', deformable_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        BottleneckX(
            64,
            64,
            groups=32,
            base_width=4,
            dcn=dcn,
            conv_cfg=dict(type='Conv'))
    BottleneckX(64, 64, dcn=dcn)

    block = BottleneckX(64, 16, groups=32, base_width=4)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_resnext_backbone():
    with pytest.raises(KeyError):
        # ResNeXt depth should be in [50, 101, 152]
        ResNeXt(depth=18)

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
