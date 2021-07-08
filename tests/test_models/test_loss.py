import pytest
import torch

from mmdet.models.losses import (BalancedL1Loss, BoundedIoULoss, CIoULoss,
                                 CrossEntropyLoss, DIoULoss,
                                 DistributionFocalLoss, FocalLoss,
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
    GaussianFocalLoss, DistributionFocalLoss, VarifocalLoss, CrossEntropyLoss
])
def test_loss_with_reduction_override(loss_class):
    pred = torch.rand((10, 4))
    target = torch.rand((10, 4))

    with pytest.raises(AssertionError):
        # only reduction_override from [None, 'none', 'mean', 'sum']
        # is not allowed
        reduction_override = True
        loss_class()(pred, target, reduction_override=reduction_override)


@pytest.mark.parametrize('loss_class', [
    IoULoss, BoundedIoULoss, GIoULoss, DIoULoss, CIoULoss, MSELoss, L1Loss,
    SmoothL1Loss, BalancedL1Loss
])
def test_regression_losses(loss_class):
    pred = torch.rand((10, 4))
    target = torch.rand((10, 4))

    # Test loss forward
    loss = loss_class()(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = loss_class()(pred, target, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = loss_class()(pred, target, avg_factor=10)
    assert isinstance(loss, torch.Tensor)

    with pytest.raises(ValueError):
        # loss can evaluate with avg_factor only if
        # reduction is None, 'none' or 'mean'.
        reduction_override = 'sum'
        loss_class()(
            pred, target, avg_factor=10, reduction_override=reduction_override)

    # Test loss forward with avg_factor and reduction
    for reduction_override in [None, 'none', 'mean']:
        loss_class()(
            pred, target, avg_factor=10, reduction_override=reduction_override)
        assert isinstance(loss, torch.Tensor)


@pytest.mark.parametrize('loss_class', [FocalLoss, CrossEntropyLoss])
def test_classification_losses(loss_class):
    pred = torch.rand((10, 5))
    target = torch.randint(0, 5, (10, ))

    # Test loss forward
    loss = loss_class()(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with reduction_override
    loss = loss_class()(pred, target, reduction_override='mean')
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with avg_factor
    loss = loss_class()(pred, target, avg_factor=10)
    assert isinstance(loss, torch.Tensor)

    with pytest.raises(ValueError):
        # loss can evaluate with avg_factor only if
        # reduction is None, 'none' or 'mean'.
        reduction_override = 'sum'
        loss_class()(
            pred, target, avg_factor=10, reduction_override=reduction_override)

    # Test loss forward with avg_factor and reduction
    for reduction_override in [None, 'none', 'mean']:
        loss_class()(
            pred, target, avg_factor=10, reduction_override=reduction_override)
        assert isinstance(loss, torch.Tensor)
