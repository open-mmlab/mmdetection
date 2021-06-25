import pytest

from mmdet.models.backbones import DetectoRS_ResNet


def test_detectorrs_resnet_backbone():
    """Test init_weights config."""
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
        DetectoRS_ResNet(
            **detectorrs_cfg, pretrained='Pretrained', init_cfg='Pretrained')
    with pytest.raises(AssertionError):
        DetectoRS_ResNet(
            **detectorrs_cfg, pretrained=None, init_cfg=['Pretrained'])
    with pytest.raises(KeyError):
        DetectoRS_ResNet(
            **detectorrs_cfg,
            pretrained=None,
            init_cfg=dict(checkpoint='Pretrained'))
    with pytest.raises(AssertionError):
        DetectoRS_ResNet(
            **detectorrs_cfg, pretrained=None, init_cfg=dict(type='Trained'))
    with pytest.raises(TypeError):
        model = DetectoRS_ResNet(
            **detectorrs_cfg, pretrained=['Pretrained'], init_cfg=None)
        model.init_weights()
