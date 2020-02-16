"""
CommandLine:
    pytest tests/test_dpool.py
"""
import torch

from mmdet.ops.dcn.deform_pool import (DeformRoIPoolingPack,
                                       ModulatedDeformRoIPoolingPack)


def test_dpool_empty_cpu():
    """
    CommandLine:
        xdoctest -m tests/test_dpool.py test_dpool_empty_cpu
    """
    dpool_layer = DeformRoIPoolingPack(
        spatial_scale=7,
        out_size=7,
        out_channels=16,
        no_trans=False,
        group_size=1,
        trans_std=0.1,
    )
    mdpool_layer = ModulatedDeformRoIPoolingPack(
        spatial_scale=7,
        out_size=7,
        out_channels=16,
        no_trans=False,
        group_size=1,
        trans_std=0.1,
    )
    dpool_trans_layer = DeformRoIPoolingPack(
        spatial_scale=7,
        out_size=7,
        out_channels=16,
        no_trans=True,
        group_size=1,
        trans_std=0.1,
    )
    mdpool_trans_layer = ModulatedDeformRoIPoolingPack(
        spatial_scale=7,
        out_size=7,
        out_channels=16,
        no_trans=True,
        group_size=1,
        trans_std=0.1,
    )

    # test whether the dpool layer can handle empty roi
    rois = torch.empty((0, 5))
    feats = torch.zeros((1, 16, 32, 32))
    data = dpool_layer(feats, rois)
    assert data.shape[0] == rois.shape[0]

    data = mdpool_layer(feats, rois)
    assert data.shape[0] == rois.shape[0]

    data = dpool_trans_layer(feats, rois)
    assert data.shape[0] == rois.shape[0]

    data = mdpool_trans_layer(feats, rois)
    assert data.shape[0] == rois.shape[0]


def test_dpool_empty_gpu():
    """
    CommandLine:
        xdoctest -m tests/test_dpool.py test_dpool_empty_cpu
    """
    dpool_layer = DeformRoIPoolingPack(
        spatial_scale=7,
        out_size=7,
        out_channels=16,
        no_trans=False,
        group_size=1,
        trans_std=0.1,
    ).cuda()
    mdpool_layer = ModulatedDeformRoIPoolingPack(
        spatial_scale=7,
        out_size=7,
        out_channels=16,
        no_trans=False,
        group_size=1,
        trans_std=0.1,
    ).cuda()
    dpool_trans_layer = DeformRoIPoolingPack(
        spatial_scale=7,
        out_size=7,
        out_channels=16,
        no_trans=True,
        group_size=1,
        trans_std=0.1,
    ).cuda()
    mdpool_trans_layer = ModulatedDeformRoIPoolingPack(
        spatial_scale=7,
        out_size=7,
        out_channels=16,
        no_trans=True,
        group_size=1,
        trans_std=0.1,
    ).cuda()

    # test whether the dpool layer can handle empty roi
    rois = torch.empty((0, 5)).cuda()
    feats = torch.zeros((1, 16, 32, 32)).cuda()
    data = dpool_layer(feats, rois)
    assert data.shape[0] == rois.shape[0]

    data = mdpool_layer(feats, rois)
    assert data.shape[0] == rois.shape[0]

    data = dpool_trans_layer(feats, rois)
    assert data.shape[0] == rois.shape[0]

    data = mdpool_trans_layer(feats, rois)
    assert data.shape[0] == rois.shape[0]
