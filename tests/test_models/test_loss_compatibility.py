import pytest
from test_forward import _demo_mm_inputs, _get_detector_cfg

from mmdet.models import build_detector


@pytest.mark.parametrize('loss_bbox', [
    dict(type='L1Loss', loss_weight=1.0),
    dict(type='GHMR', mu=0.02, bins=10, momentum=0.7, loss_weight=10.0),
    dict(type='IoULoss', loss_weight=1.0),
    dict(type='BoundedIoULoss', loss_weight=1.0),
    dict(type='GIoULoss', loss_weight=1.0),
    dict(type='DIoULoss', loss_weight=1.0),
    dict(type='CIoULoss', loss_weight=1.0),
    dict(type='MSELoss', loss_weight=1.0),
    dict(type='SmoothL1Loss', loss_weight=1.0),
    dict(type='BalancedL1Loss', loss_weight=1.0)
])
def test_bbox_loss_compatibility(loss_bbox):
    """Test loss_bbox compatibility.

    Using Faster R-CNN as a sample, modifying the loss function in the config
    file to verify the compatibility of Loss APIS
    """
    # Faster R-CNN config dict
    config_path = '_base_/models/faster_rcnn_r50_fpn.py'
    cfg_model = _get_detector_cfg(config_path)

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    if 'IoULoss' in loss_bbox['type']:
        cfg_model.roi_head.bbox_head.reg_decoded_bbox = True

    cfg_model.roi_head.bbox_head.loss_bbox = loss_bbox
    detector = build_detector(cfg_model)
    loss = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(loss, dict)
    loss, _ = detector._parse_losses(loss)
    assert float(loss.item()) > 0


@pytest.mark.parametrize('loss_cls', [
    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=1.0),
    dict(
        type='GHMC', bins=30, momentum=0.75, use_sigmoid=True, loss_weight=1.0)
])
def test_cls_loss_compatibility(loss_cls):
    """Test loss_cls compatibility.

    Using Faster R-CNN as a sample, modifying the loss function in the config
    file to verify the compatibility of Loss APIS
    """
    # Faster R-CNN config dict
    config_path = '_base_/models/faster_rcnn_r50_fpn.py'
    cfg_model = _get_detector_cfg(config_path)

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # verify class loss function compatibility
    # for loss_cls in loss_clses:
    cfg_model.roi_head.bbox_head.loss_cls = loss_cls
    detector = build_detector(cfg_model)
    loss = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(loss, dict)
    loss, _ = detector._parse_losses(loss)
    assert float(loss.item()) > 0
