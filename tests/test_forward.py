"""
pytest tests/test_forward.py
"""
import copy
from os.path import dirname, exists, join

import numpy as np
import torch


def _get_config_directory():
    """ Find the predefined detector config directory """
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """
    Load a configuration as a python module
    """
    from xdoctest.utils import import_module_from_path
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = import_module_from_path(config_fpath)
    return config_mod


def _get_detector_cfg(fname):
    """
    Grab configs necessary to create a detector. These are deep copied to allow
    for safe modification of parameters without influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))
    return model, train_cfg, test_cfg


def test_ssd300_forward():
    model, train_cfg, test_cfg = _get_detector_cfg('ssd300_coco.py')
    model['pretrained'] = None

    from mmdet.models import build_detector
    detector = build_detector(model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 3, 300, 300)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)


def test_rpn_forward():
    model, train_cfg, test_cfg = _get_detector_cfg('rpn_r50_fpn_1x.py')
    model['pretrained'] = None

    from mmdet.models import build_detector
    detector = build_detector(model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 3, 224, 224)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    gt_bboxes = mm_inputs['gt_bboxes']
    losses = detector.forward(
        imgs, img_metas, gt_bboxes=gt_bboxes, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)


def test_retina_ghm_forward():
    model, train_cfg, test_cfg = _get_detector_cfg(
        'ghm/retinanet_ghm_r50_fpn_1x.py')
    model['pretrained'] = None

    from mmdet.models import build_detector
    detector = build_detector(model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (3, 3, 224, 224)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)

    if torch.cuda.is_available():
        detector = detector.cuda()
        imgs = imgs.cuda()
        # Test forward train
        gt_bboxes = [b.cuda() for b in mm_inputs['gt_bboxes']]
        gt_labels = [g.cuda() for g in mm_inputs['gt_labels']]
        losses = detector.forward(
            imgs,
            img_metas,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            return_loss=True)
        assert isinstance(losses, dict)

        # Test forward test
        with torch.no_grad():
            img_list = [g[None, :] for g in imgs]
            batch_results = []
            for one_img, one_meta in zip(img_list, img_metas):
                result = detector.forward([one_img], [[one_meta]],
                                          return_loss=False)
                batch_results.append(result)


def _demo_mm_inputs(input_shape=(1, 3, 300, 300),
                    num_items=None, num_classes=10):  # yapf: disable
    """
    Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_items (None | List[int]):
            specifies the number of boxes in each batch item

        num_classes (int):
            number of different labels a box might have
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    } for _ in range(N)]

    gt_bboxes = []
    gt_labels = []

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

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
    }
    return mm_inputs
