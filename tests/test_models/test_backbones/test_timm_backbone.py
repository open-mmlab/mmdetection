# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet.models.backbones import TIMMBackbone
from .utils import check_norm_state


def test_timm_backbone():
    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = TIMMBackbone()
        model.init_weights(pretrained=0)

    # Test different norm_layer, can be: 'SyncBN', 'BN2d', 'GN', 'LN', 'IN'
    # Test resnet18 from timm, norm_layer='BN2d'
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        norm_layer='BN2d')

    # Test resnet18 from timm, norm_layer='SyncBN'
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        norm_layer='SyncBN')

    # Test resnet18 from timm, features_only=True, output_stride=32
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 112, 112))
    assert feats[1].shape == torch.Size((1, 64, 56, 56))
    assert feats[2].shape == torch.Size((1, 128, 28, 28))
    assert feats[3].shape == torch.Size((1, 256, 14, 14))
    assert feats[4].shape == torch.Size((1, 512, 7, 7))

    # Test resnet18 from timm, features_only=True, output_stride=32,
    # out_indices=(1, 2, 3)
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=32,
        out_indices=(1, 2, 3))
    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    assert len(feats) == 3
    assert feats[0].shape == torch.Size((1, 64, 56, 56))
    assert feats[1].shape == torch.Size((1, 128, 28, 28))
    assert feats[2].shape == torch.Size((1, 256, 14, 14))

    # Test resnet18 from timm, features_only=True, output_stride=16
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=16)
    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 112, 112))
    assert feats[1].shape == torch.Size((1, 64, 56, 56))
    assert feats[2].shape == torch.Size((1, 128, 28, 28))
    assert feats[3].shape == torch.Size((1, 256, 14, 14))
    assert feats[4].shape == torch.Size((1, 512, 14, 14))

    # Test resnet18 from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnet18',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 224, 224)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 112, 112))
    assert feats[1].shape == torch.Size((1, 64, 56, 56))
    assert feats[2].shape == torch.Size((1, 128, 28, 28))
    assert feats[3].shape == torch.Size((1, 256, 28, 28))
    assert feats[4].shape == torch.Size((1, 512, 28, 28))

    # Test efficientnet_b1 with pretrained weights
    model = TIMMBackbone(model_name='efficientnet_b1', pretrained=True)

    # Test resnetv2_50x1_bitm from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_50x1_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 8, 8)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 4, 4))
    assert feats[1].shape == torch.Size((1, 256, 2, 2))
    assert feats[2].shape == torch.Size((1, 512, 1, 1))
    assert feats[3].shape == torch.Size((1, 1024, 1, 1))
    assert feats[4].shape == torch.Size((1, 2048, 1, 1))

    # Test resnetv2_50x3_bitm from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_50x3_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 8, 8)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 192, 4, 4))
    assert feats[1].shape == torch.Size((1, 768, 2, 2))
    assert feats[2].shape == torch.Size((1, 1536, 1, 1))
    assert feats[3].shape == torch.Size((1, 3072, 1, 1))
    assert feats[4].shape == torch.Size((1, 6144, 1, 1))

    # Test resnetv2_101x1_bitm from timm, features_only=True, output_stride=8
    model = TIMMBackbone(
        model_name='resnetv2_101x1_bitm',
        features_only=True,
        pretrained=False,
        output_stride=8)
    imgs = torch.randn(1, 3, 8, 8)
    feats = model(imgs)
    assert len(feats) == 5
    assert feats[0].shape == torch.Size((1, 64, 4, 4))
    assert feats[1].shape == torch.Size((1, 256, 2, 2))
    assert feats[2].shape == torch.Size((1, 512, 1, 1))
    assert feats[3].shape == torch.Size((1, 1024, 1, 1))
    assert feats[4].shape == torch.Size((1, 2048, 1, 1))
