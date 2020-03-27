import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones import ResNet, ResNetV1d, ResNeXt
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.models.backbones.resnext import Bottleneck as BottleneckX
from mmdet.models.utils import ResLayer


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
    x = torch.randn(2, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 64, 56, 56])


def test_resnet_bottleneck():

    with pytest.raises(AssertionError):
        # style must be in ['pytorch', 'caffe']
        Bottleneck(64, 64, style='tensorflow')

    block = Bottleneck(64, 64, stride=2, style='pytorch')
    assert block.conv1.stride == (1, 1)
    assert block.conv2.stride == (2, 2)
    block = Bottleneck(64, 64, stride=2, style='caffe')
    assert block.conv1.stride == (2, 2)
    assert block.conv2.stride == (1, 1)

    dcn = dict(type='DCN', deformable_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        Bottleneck(64, 64, dcn=dcn, conv_cfg=dict(type='Conv'))
    Bottleneck(64, 64, dcn=dcn)

    block = Bottleneck(64, 16)
    x = torch.randn(2, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 64, 56, 56])

    gcb = dict(ratio=1. / 4., )
    block = Bottleneck(64, 16, gcb=gcb)
    x = torch.randn(2, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 64, 56, 56])

    gen_attention = dict(
        spatial_range=-1, num_heads=8, attention_type='0010', kv_stride=2)
    block = Bottleneck(64, 16, gen_attention=gen_attention)
    x = torch.randn(2, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 64, 56, 56])


def test_resnet_res_layer():
    layer = ResLayer(Bottleneck, 64, 16, 3)
    assert len(layer) == 3
    assert layer[0].conv1.in_channels == 64
    assert layer[0].conv1.out_channels == 16
    for i in range(len(layer)):
        assert layer[i].downsample is None

    layer = ResLayer(Bottleneck, 64, 64, 3)
    assert layer[0].downsample[0].out_channels == 256
    for i in range(1, len(layer)):
        assert layer[i].downsample is None

    layer = ResLayer(Bottleneck, 64, 64, 3, stride=2)
    for i in range(len(layer)):
        if layer[i].downsample is not None:
            assert layer[i].downsample[0].stride == (2, 2)

    layer = ResLayer(Bottleneck, 64, 64, 3, stride=2, avg_down=True)
    for i in range(len(layer)):
        if layer[i].downsample is not None:
            assert layer[i].downsample[1].stride == (1, 1)


def test_resnet_backbone():
    """Test resnet backbone"""
    with pytest.raises(KeyError):
        # ResNet depth should be in [18, 34, 50, 101, 152]
        ResNet(20)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=0)

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

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert feat[0].shape == torch.Size([2, 64, 56, 56])
    assert feat[1].shape == torch.Size([2, 128, 28, 28])
    assert feat[2].shape == torch.Size([2, 256, 14, 14])
    assert feat[3].shape == torch.Size([2, 512, 7, 7])

    model = ResNet(50)
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert feat[0].shape == torch.Size([2, 256, 56, 56])
    assert feat[1].shape == torch.Size([2, 512, 28, 28])
    assert feat[2].shape == torch.Size([2, 1024, 14, 14])
    assert feat[3].shape == torch.Size([2, 2048, 7, 7])

    model = ResNetV1d(depth=50)
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert feat[0].shape == torch.Size([2, 256, 56, 56])
    assert feat[1].shape == torch.Size([2, 512, 28, 28])
    assert feat[2].shape == torch.Size([2, 1024, 14, 14])
    assert feat[3].shape == torch.Size([2, 2048, 7, 7])


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
    x = torch.randn(2, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 64, 56, 56])


def test_resnext_backbone():
    model = ResNeXt(depth=50, groups=32, base_width=4)
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert feat[0].shape == torch.Size([2, 256, 56, 56])
    assert feat[1].shape == torch.Size([2, 512, 28, 28])
    assert feat[2].shape == torch.Size([2, 1024, 14, 14])
    assert feat[3].shape == torch.Size([2, 2048, 7, 7])
