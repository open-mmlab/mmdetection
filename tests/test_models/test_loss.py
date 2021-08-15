import pytest
import torch

from mmdet.models.losses import (BalancedL1Loss, CrossEntropyLoss,
                                 DistributionFocalLoss, FocalLoss,
                                 GaussianFocalLoss,
                                 KnowledgeDistillationKLDivLoss, L1Loss,
                                 MSELoss, QualityFocalLoss, SeesawLoss,
                                 SmoothL1Loss, VarifocalLoss)
from mmdet.models.losses.ghm_loss import GHMC, GHMR
from mmdet.models.losses.iou_loss import (BoundedIoULoss, CIoULoss, DIoULoss,
                                          GIoULoss, IoULoss)


@pytest.mark.parametrize(
    'loss_class', [IoULoss, BoundedIoULoss, GIoULoss, DIoULoss, CIoULoss])
def test_iou_type_loss_zeros_weight(loss_class):
    pred = torch.rand((10, 4))
    target = torch.rand((10, 4))
    weight = torch.zeros(10)

    loss = loss_class()(pred, target, weight)
    assert loss == 0.


@pytest.mark.parametrize('loss_class', [
    BalancedL1Loss, BoundedIoULoss, CIoULoss, CrossEntropyLoss, DIoULoss,
    FocalLoss, DistributionFocalLoss, MSELoss, SeesawLoss, GaussianFocalLoss,
    GIoULoss, IoULoss, L1Loss, QualityFocalLoss, VarifocalLoss, GHMR, GHMC,
    SmoothL1Loss, KnowledgeDistillationKLDivLoss
])
def test_loss_with_reduction_override(loss_class):
    pred = torch.rand((10, 4))
    target = torch.rand((10, 4)),
    weight = None

    with pytest.raises(AssertionError):
        # only reduction_override from [None, 'none', 'mean', 'sum']
        # is not allowed
        reduction_override = True
        loss_class()(
            pred, target, weight, reduction_override=reduction_override)


@pytest.mark.parametrize('loss_class', [
    IoULoss, BoundedIoULoss, GIoULoss, DIoULoss, CIoULoss, MSELoss, L1Loss,
    SmoothL1Loss, BalancedL1Loss
])
def test_regression_losses(loss_class):
    pred = torch.rand((10, 4))
    target = torch.rand((10, 4))
    weight = torch.rand((10, 4))

    # Test loss forward
    loss = loss_class()(pred, target)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with weight
    loss = loss_class()(pred, target, weight)
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


@pytest.mark.parametrize('loss_class', [GHMR])
def test_GHMR_loss(loss_class):
    pred = torch.rand((10, 4))
    target = torch.rand((10, 4))
    weight = torch.rand((10, 4))

    # Test loss forward
    loss = loss_class()(pred, target, weight)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.parametrize('use_sigmoid', [True, False])
def test_loss_with_ignore_index(use_sigmoid):
    # Test cross_entropy loss
    loss_class = CrossEntropyLoss(
        use_sigmoid=use_sigmoid, use_mask=False, ignore_index=255)
    pred = torch.rand((10, 5))
    target = torch.randint(0, 5, (10, ))

    ignored_indices = torch.randint(0, 10, (2, ), dtype=torch.long)
    target[ignored_indices] = 255

    # Test loss forward with default ignore
    loss_with_ignore = loss_class(pred, target, reduction_override='sum')
    assert isinstance(loss_with_ignore, torch.Tensor)

    # Test loss forward with forward ignore
    target[ignored_indices] = 250
    loss_with_forward_ignore = loss_class(
        pred, target, ignore_index=250, reduction_override='sum')
    assert isinstance(loss_with_forward_ignore, torch.Tensor)

    # Verify correctness
    not_ignored_indices = (target != 250)
    pred = pred[not_ignored_indices]
    target = target[not_ignored_indices]
    loss = loss_class(pred, target, reduction_override='sum')

    assert torch.allclose(loss, loss_with_ignore)
    assert torch.allclose(loss, loss_with_forward_ignore)
