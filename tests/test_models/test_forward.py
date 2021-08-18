"""pytest tests/test_forward.py."""
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


def test_sparse_rcnn_forward():
    config_path = 'sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py'
    model = _get_detector_cfg(config_path)
    model.backbone.init_cfg = None
    from mmdet.models import build_detector
    detector = build_detector(model)
    detector.init_weights()
    input_shape = (1, 3, 550, 550)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[5])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    # Test forward train with non-empty truth batch
    detector.train()
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_bboxes = [item for item in gt_bboxes]
    gt_labels = mm_inputs['gt_labels']
    gt_labels = [item for item in gt_labels]
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0
    detector.forward_dummy(imgs)

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_bboxes = [item for item in gt_bboxes]
    gt_labels = mm_inputs['gt_labels']
    gt_labels = [item for item in gt_labels]
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0

    # Test forward test
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      rescale=True,
                                      return_loss=False)
            batch_results.append(result)

    # test empty proposal in roi_head
    with torch.no_grad():
        # test no proposal in the whole batch
        detector.roi_head.simple_test([imgs[0][None, :]], torch.empty(
            (1, 0, 4)), torch.empty((1, 100, 4)), [img_metas[0]],
                                      torch.ones((1, 4)))


def test_rpn_forward():
    model = _get_detector_cfg('rpn/rpn_r50_fpn_1x_coco.py')
    model.backbone.init_cfg = None

    from mmdet.models import build_detector
    detector = build_detector(model)

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


@pytest.mark.parametrize(
    'cfg_file',
    [
        'retinanet/retinanet_r50_fpn_1x_coco.py',
        'guided_anchoring/ga_retinanet_r50_fpn_1x_coco.py',
        'ghm/retinanet_ghm_r50_fpn_1x_coco.py',
        'fcos/fcos_center_r50_caffe_fpn_gn-head_1x_coco.py',
        'foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py',
        # 'free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py',
        # 'atss/atss_r50_fpn_1x_coco.py',  # not ready for topk
        'reppoints/reppoints_moment_r50_fpn_1x_coco.py',
        'yolo/yolov3_d53_mstrain-608_273e_coco.py'
    ])
def test_single_stage_forward_gpu(cfg_file):
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')

    model = _get_detector_cfg(cfg_file)
    model.backbone.init_cfg = None

    from mmdet.models import build_detector
    detector = build_detector(model)

    input_shape = (2, 3, 224, 224)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

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


def test_faster_rcnn_ohem_forward():
    model = _get_detector_cfg(
        'faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py')
    model.backbone.init_cfg = None

    from mmdet.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 256, 256)

    # Test forward train with a non-empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0


@pytest.mark.parametrize(
    'cfg_file',
    [
        'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py',
        'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py',
        'grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py',
        'ms_rcnn/ms_rcnn_r50_fpn_1x_coco.py',
        'htc/htc_r50_fpn_1x_coco.py',
        'panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py',
        'scnet/scnet_r50_fpn_20e_coco.py',
        'seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py'  # noqa: E501
    ])
def test_two_stage_forward(cfg_file):
    models_with_semantic = [
        'htc/htc_r50_fpn_1x_coco.py',
        'panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco.py',
        'scnet/scnet_r50_fpn_20e_coco.py',
    ]
    if cfg_file in models_with_semantic:
        with_semantic = True
    else:
        with_semantic = False

    model = _get_detector_cfg(cfg_file)
    model.backbone.init_cfg = None

    # Save cost
    if cfg_file in [
            'seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py'  # noqa: E501
    ]:
        model.roi_head.bbox_head.num_classes = 80
        model.roi_head.bbox_head.loss_cls.num_classes = 80
        model.roi_head.mask_head.num_classes = 80
        model.test_cfg.rcnn.score_thr = 0.05
        model.test_cfg.rcnn.max_per_img = 100

    from mmdet.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 256, 256)

    # Test forward train with a non-empty truth batch
    mm_inputs = _demo_mm_inputs(
        input_shape, num_items=[10], with_semantic=with_semantic)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    losses = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(
        input_shape, num_items=[0], with_semantic=with_semantic)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    losses = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    loss.requires_grad_(True)
    assert float(loss.item()) > 0
    loss.backward()

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      return_loss=False)
            batch_results.append(result)
    cascade_models = [
        'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py',
        'htc/htc_r50_fpn_1x_coco.py',
        'scnet/scnet_r50_fpn_20e_coco.py',
    ]
    # test empty proposal in roi_head
    with torch.no_grad():
        # test no proposal in the whole batch
        detector.simple_test(
            imgs[0][None, :], [img_metas[0]], proposals=[torch.empty((0, 4))])

        # test no proposal of aug
        features = detector.extract_feats([imgs[0][None, :]] * 2)
        detector.roi_head.aug_test(features, [torch.empty((0, 4))] * 2,
                                   [[img_metas[0]]] * 2)

        # test rcnn_test_cfg is None
        if cfg_file not in cascade_models:
            feature = detector.extract_feat(imgs[0][None, :])
            bboxes, scores = detector.roi_head.simple_test_bboxes(
                feature, [img_metas[0]], [torch.empty((0, 4))], None)
            assert all([bbox.shape == torch.Size((0, 4)) for bbox in bboxes])
            assert all([
                score.shape == torch.Size(
                    (0, detector.roi_head.bbox_head.fc_cls.out_features))
                for score in scores
            ])

        # test no proposal in the some image
        x1y1 = torch.randint(1, 100, (10, 2)).float()
        # x2y2 must be greater than x1y1
        x2y2 = x1y1 + torch.randint(1, 100, (10, 2))
        detector.simple_test(
            imgs[0][None, :].repeat(2, 1, 1, 1), [img_metas[0]] * 2,
            proposals=[torch.empty((0, 4)),
                       torch.cat([x1y1, x2y2], dim=-1)])

        # test no proposal of aug
        detector.roi_head.aug_test(
            features, [torch.cat([x1y1, x2y2], dim=-1),
                       torch.empty((0, 4))], [[img_metas[0]]] * 2)

        # test rcnn_test_cfg is None
        if cfg_file not in cascade_models:
            feature = detector.extract_feat(imgs[0][None, :].repeat(
                2, 1, 1, 1))
            bboxes, scores = detector.roi_head.simple_test_bboxes(
                feature, [img_metas[0]] * 2,
                [torch.empty((0, 4)),
                 torch.cat([x1y1, x2y2], dim=-1)], None)
            assert bboxes[0].shape == torch.Size((0, 4))
            assert scores[0].shape == torch.Size(
                (0, detector.roi_head.bbox_head.fc_cls.out_features))


@pytest.mark.parametrize(
    'cfg_file', ['ghm/retinanet_ghm_r50_fpn_1x_coco.py', 'ssd/ssd300_coco.py'])
def test_single_stage_forward_cpu(cfg_file):
    model = _get_detector_cfg(cfg_file)
    model.backbone.init_cfg = None

    from mmdet.models import build_detector
    detector = build_detector(model)

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


def test_yolact_forward():
    model = _get_detector_cfg('yolact/yolact_r50_1x8_coco.py')
    model.backbone.init_cfg = None

    from mmdet.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 100, 100)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    detector.train()
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    gt_masks = mm_inputs['gt_masks']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      rescale=True,
                                      return_loss=False)
            batch_results.append(result)


def test_detr_forward():
    model = _get_detector_cfg('detr/detr_r50_8x2_150e_coco.py')
    model.backbone.init_cfg = None

    from mmdet.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 100, 100)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train with non-empty truth batch
    detector.train()
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0

    # Test forward test
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      rescale=True,
                                      return_loss=False)
            batch_results.append(result)


def test_kd_single_stage_forward():
    model = _get_detector_cfg('ld/ld_r18_gflv1_r101_fpn_coco_1x.py')
    model.backbone.init_cfg = None

    from mmdet.models import build_detector
    detector = build_detector(model)

    input_shape = (1, 3, 100, 100)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train with non-empty truth batch
    detector.train()
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0

    # Test forward train with an empty truth batch
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[0])
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    gt_bboxes = mm_inputs['gt_bboxes']
    gt_labels = mm_inputs['gt_labels']
    losses = detector.forward(
        imgs,
        img_metas,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        return_loss=True)
    assert isinstance(losses, dict)
    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) > 0

    # Test forward test
    detector.eval()
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img, one_meta in zip(img_list, img_metas):
            result = detector.forward([one_img], [[one_meta]],
                                      rescale=True,
                                      return_loss=False)
            batch_results.append(result)


def test_inference_detector():
    from mmdet.apis import inference_detector
    from mmdet.models import build_detector
    from mmcv import ConfigDict

    # small RetinaNet
    num_class = 3
    model_dict = dict(
        type='RetinaNet',
        backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch'),
        neck=None,
        bbox_head=dict(
            type='RetinaHead',
            num_classes=num_class,
            in_channels=512,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5],
                strides=[32]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
        ),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))

    rng = np.random.RandomState(0)
    img1 = rng.rand(100, 100, 3)
    img2 = rng.rand(100, 100, 3)

    model = build_detector(ConfigDict(model_dict))
    config = _get_config_module('retinanet/retinanet_r50_fpn_1x_coco.py')
    model.cfg = config
    # test single image
    result = inference_detector(model, img1)
    assert len(result) == num_class
    # test multiple image
    result = inference_detector(model, [img1, img2])
    assert len(result) == 2 and len(result[0]) == num_class
