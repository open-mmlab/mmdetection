# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmdet.models.backbones import DetectoRS_ResNet


def test_detectorrs_resnet_backbone():
    detectorrs_cfg = dict(
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True)
    """Test init_weights config"""
    with pytest.raises(AssertionError):
        # pretrained and init_cfg cannot be specified at the same time
        DetectoRS_ResNet(
            **detectorrs_cfg, pretrained='Pretrained', init_cfg='Pretrained')

    with pytest.raises(AssertionError):
        # init_cfg must be a dict
        DetectoRS_ResNet(
            **detectorrs_cfg, pretrained=None, init_cfg=['Pretrained'])

    with pytest.raises(KeyError):
        # init_cfg must contain the key `type`
        DetectoRS_ResNet(
            **detectorrs_cfg,
            pretrained=None,
            init_cfg=dict(checkpoint='Pretrained'))

    with pytest.raises(AssertionError):
        # init_cfg only support initialize pretrained model way
        DetectoRS_ResNet(
            **detectorrs_cfg, pretrained=None, init_cfg=dict(type='Trained'))

    with pytest.raises(TypeError):
        # pretrained mast be a str or None
        model = DetectoRS_ResNet(
            **detectorrs_cfg, pretrained=['Pretrained'], init_cfg=None)
        model.init_weights()
