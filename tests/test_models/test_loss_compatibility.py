# Copyright (c) OpenMMLab. All rights reserved.
"""pytest tests/test_loss_compatibility.py."""
import copy
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmcv import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


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

    from mmdet.models import build_detector
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

    from mmdet.models import build_detector
    detector = build_detector(cfg_model)

    loss = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(loss, dict)
    loss, _ = detector._parse_losses(loss)
    assert float(loss.item()) > 0


def _demo_mm_inputs(input_shape=(1, 3, 300, 300),
                    num_items=None, num_classes=10,
                    with_semantic=False):  # yapf: disable
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_items (None | List[int]):
            specifies the number of boxes in each batch item

        num_classes (int):
            number of different labels a box might have
    """
    from mmdet.core import BitmapMasks

    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': np.array([1.1, 1.2, 1.1, 1.2]),
        'flip': False,
        'flip_direction': None,
    } for _ in range(N)]

    gt_bboxes = []
    gt_labels = []
    gt_masks = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
        tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
        br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
        br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))

    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(BitmapMasks(mask, H, W))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
        'gt_masks': gt_masks,
    }

    if with_semantic:
        # assume gt_semantic_seg using scale 1/8 of the img
        gt_semantic_seg = np.random.randint(
            0, num_classes, (1, 1, H // 8, W // 8), dtype=np.uint8)
        mm_inputs.update(
            {'gt_semantic_seg': torch.ByteTensor(gt_semantic_seg)})

    return mm_inputs
