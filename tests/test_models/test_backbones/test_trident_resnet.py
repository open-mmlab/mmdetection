import pytest
import torch

from mmdet.models.backbones import TridentResNet
from mmdet.models.backbones.trident_resnet import TridentBottleneck


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
