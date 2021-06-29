import pytest
import torch

from mmdet.models.losses import (BalancedL1Loss, BoundedIoULoss, CIoULoss,
                                 DIoULoss, DistributionFocalLoss, FocalLoss,
                                 GaussianFocalLoss, GIoULoss, IoULoss, L1Loss,
                                 MSELoss, QualityFocalLoss, SmoothL1Loss,
                                 VarifocalLoss)


@pytest.mark.parametrize(
    'loss_class', [IoULoss, BoundedIoULoss, GIoULoss, DIoULoss, CIoULoss])
def test_iou_type_loss_zeros_weight(loss_class):
    pred = torch.rand((10, 4))
    target = torch.rand((10, 4))
    weight = torch.zeros(10)

    loss = loss_class()(pred, target, weight)
    assert loss == 0.


@pytest.mark.parametrize('loss_class', [
    IoULoss, BoundedIoULoss, GIoULoss, DIoULoss, CIoULoss, MSELoss, L1Loss,
    SmoothL1Loss, BalancedL1Loss, FocalLoss, QualityFocalLoss,
    GaussianFocalLoss, DistributionFocalLoss, VarifocalLoss
])
def test_loss_with_reduction_override(loss_class):
    pred = torch.rand((10, 4))
    target = torch.rand((10, 4))

    with pytest.raises(AssertionError):
        # assert whether reduction_override in loss function
        reduction_override = True
        loss_class()(pred, target, reduction_override=reduction_override)
