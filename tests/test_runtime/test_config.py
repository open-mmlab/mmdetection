from os.path import dirname, exists, join, relpath
from unittest.mock import Mock

import pytest
import torch
from mmcv.runner import build_optimizer

from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.utils import NumClassCheckHook


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(__file__))
        repo_dpath = join(repo_dpath, '..')
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _check_numclasscheckhook(detector, config_mod):
    dummy_runner = Mock()
    dummy_runner.model = detector

    def get_dataset_name_classes(dataset):
        # deal with `RepeatDataset`,`ConcatDataset`,`ClassBalancedDataset`..
        if isinstance(dataset, (list, tuple)):
            dataset = dataset[0]
        while ('dataset' in dataset):
            dataset = dataset['dataset']
            # ConcatDataset
            if isinstance(dataset, (list, tuple)):
                dataset = dataset[0]
        return dataset['type'], dataset.get('classes', None)

    compatible_check = NumClassCheckHook()
    dataset_name, CLASSES = get_dataset_name_classes(
        config_mod['data']['train'])
    if CLASSES is None:
        CLASSES = DATASETS.get(dataset_name).CLASSES
    dummy_runner.data_loader.dataset.CLASSES = CLASSES
    compatible_check.before_train_epoch(dummy_runner)

    dummy_runner.data_loader.dataset.CLASSES = None
    compatible_check.before_train_epoch(dummy_runner)

    dataset_name, CLASSES = get_dataset_name_classes(config_mod['data']['val'])
    if CLASSES is None:
        CLASSES = DATASETS.get(dataset_name).CLASSES
    dummy_runner.data_loader.dataset.CLASSES = CLASSES
    compatible_check.before_val_epoch(dummy_runner)
    dummy_runner.data_loader.dataset.CLASSES = None
    compatible_check.before_val_epoch(dummy_runner)


def test_config_build_detector():
    """Test that all detection models defined in the configs can be
    initialized."""
    from mmcv import Config
    from mmdet.models import build_detector

    config_dpath = _get_config_directory()
    print(f'Found config_dpath = {config_dpath}')

    import glob
    config_fpaths = list(glob.glob(join(config_dpath, '**', '*.py')))
    config_fpaths = [p for p in config_fpaths if p.find('_base_') == -1]
    config_names = [relpath(p, config_dpath) for p in config_fpaths]

    print(f'Using {len(config_names)} config files')

    for config_fname in config_names:
        config_fpath = join(config_dpath, config_fname)
        config_mod = Config.fromfile(config_fpath)
        config_mod.model
        print(f'Building detector, config_fpath = {config_fpath}')

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model:
            config_mod.model['pretrained'] = None

        detector = build_detector(config_mod.model)
        assert detector is not None

        _check_numclasscheckhook(detector, config_mod)

        optimizer = build_optimizer(detector, config_mod.optimizer)
        assert isinstance(optimizer, torch.optim.Optimizer)

        if 'roi_head' in config_mod.model.keys():
            # for two stage detector
            # detectors must have bbox head
            assert detector.roi_head.with_bbox and detector.with_bbox
            assert detector.roi_head.with_mask == detector.with_mask

            head_config = config_mod.model['roi_head']
            _check_roi_head(head_config, detector.roi_head)

        # else:
        #     # for single stage detector
        #     # detectors must have bbox head
        #     # assert detector.with_bbox
        #     head_config = config_mod.model['bbox_head']
        #     _check_bbox_head(head_config, detector.bbox_head)


def _check_roi_head(config, head):
    # check consistency between head_config and roi_head
    assert config['type'] == head.__class__.__name__

    # check roi_align
    bbox_roi_cfg = config.bbox_roi_extractor
    bbox_roi_extractor = head.bbox_roi_extractor
    _check_roi_extractor(bbox_roi_cfg, bbox_roi_extractor)

    # check bbox head infos
    bbox_cfg = config.bbox_head
    bbox_head = head.bbox_head
    _check_bbox_head(bbox_cfg, bbox_head)

    if head.with_mask:
        # check roi_align
        if config.mask_roi_extractor:
            mask_roi_cfg = config.mask_roi_extractor
            mask_roi_extractor = head.mask_roi_extractor
            _check_roi_extractor(mask_roi_cfg, mask_roi_extractor,
                                 bbox_roi_extractor)

        # check mask head infos
        mask_head = head.mask_head
        mask_cfg = config.mask_head
        _check_mask_head(mask_cfg, mask_head)

    # check arch specific settings, e.g., cascade/htc
    if config['type'] in ['CascadeRoIHead', 'HybridTaskCascadeRoIHead']:
        assert config.num_stages == len(head.bbox_head)
        assert config.num_stages == len(head.bbox_roi_extractor)

        if head.with_mask:
            assert config.num_stages == len(head.mask_head)
            assert config.num_stages == len(head.mask_roi_extractor)

    elif config['type'] in ['MaskScoringRoIHead']:
        assert (hasattr(head, 'mask_iou_head')
                and head.mask_iou_head is not None)
        mask_iou_cfg = config.mask_iou_head
        mask_iou_head = head.mask_iou_head
        assert (mask_iou_cfg.fc_out_channels ==
                mask_iou_head.fc_mask_iou.in_features)

    elif config['type'] in ['GridRoIHead']:
        grid_roi_cfg = config.grid_roi_extractor
        grid_roi_extractor = head.grid_roi_extractor
        _check_roi_extractor(grid_roi_cfg, grid_roi_extractor,
                             bbox_roi_extractor)

        config.grid_head.grid_points = head.grid_head.grid_points


def _check_roi_extractor(config, roi_extractor, prev_roi_extractor=None):
    import torch.nn as nn
    # Separate roi_extractor and prev_roi_extractor checks for flexibility
    if isinstance(roi_extractor, nn.ModuleList):
        roi_extractor = roi_extractor[0]
    if prev_roi_extractor and isinstance(prev_roi_extractor, nn.ModuleList):
        prev_roi_extractor = prev_roi_extractor[0]

    assert (len(config.featmap_strides) == len(roi_extractor.roi_layers))
    assert (config.out_channels == roi_extractor.out_channels)
    from torch.nn.modules.utils import _pair
    assert (_pair(config.roi_layer.output_size) ==
            roi_extractor.roi_layers[0].output_size)

    if 'use_torchvision' in config.roi_layer:
        assert (config.roi_layer.use_torchvision ==
                roi_extractor.roi_layers[0].use_torchvision)
    elif 'aligned' in config.roi_layer:
        assert (
            config.roi_layer.aligned == roi_extractor.roi_layers[0].aligned)

    if prev_roi_extractor:
        assert (roi_extractor.roi_layers[0].aligned ==
                prev_roi_extractor.roi_layers[0].aligned)
        assert (roi_extractor.roi_layers[0].use_torchvision ==
                prev_roi_extractor.roi_layers[0].use_torchvision)


def _check_mask_head(mask_cfg, mask_head):
    import torch.nn as nn
    if isinstance(mask_cfg, list):
        for single_mask_cfg, single_mask_head in zip(mask_cfg, mask_head):
            _check_mask_head(single_mask_cfg, single_mask_head)
    elif isinstance(mask_head, nn.ModuleList):
        for single_mask_head in mask_head:
            _check_mask_head(mask_cfg, single_mask_head)
    else:
        assert mask_cfg['type'] == mask_head.__class__.__name__
        assert mask_cfg.in_channels == mask_head.in_channels
        class_agnostic = mask_cfg.get('class_agnostic', False)
        out_dim = (1 if class_agnostic else mask_cfg.num_classes)
        if hasattr(mask_head, 'conv_logits'):
            assert (mask_cfg.conv_out_channels ==
                    mask_head.conv_logits.in_channels)
            assert mask_head.conv_logits.out_channels == out_dim
        else:
            assert mask_cfg.fc_out_channels == mask_head.fc_logits.in_features
            assert (mask_head.fc_logits.out_features == out_dim *
                    mask_head.output_area)


def _check_bbox_head(bbox_cfg, bbox_head):
    import torch.nn as nn
    if isinstance(bbox_cfg, list):
        for single_bbox_cfg, single_bbox_head in zip(bbox_cfg, bbox_head):
            _check_bbox_head(single_bbox_cfg, single_bbox_head)
    elif isinstance(bbox_head, nn.ModuleList):
        for single_bbox_head in bbox_head:
            _check_bbox_head(bbox_cfg, single_bbox_head)
    else:
        assert bbox_cfg['type'] == bbox_head.__class__.__name__
        if bbox_cfg['type'] == 'SABLHead':
            assert bbox_cfg.cls_in_channels == bbox_head.cls_in_channels
            assert bbox_cfg.reg_in_channels == bbox_head.reg_in_channels

            cls_out_channels = bbox_cfg.get('cls_out_channels', 1024)
            assert (cls_out_channels == bbox_head.fc_cls.in_features)
            assert (bbox_cfg.num_classes + 1 == bbox_head.fc_cls.out_features)

        elif bbox_cfg['type'] == 'DIIHead':
            assert bbox_cfg['num_ffn_fcs'] == bbox_head.ffn.num_fcs
            # 3 means FC and LN and Relu
            assert bbox_cfg['num_cls_fcs'] == len(bbox_head.cls_fcs) // 3
            assert bbox_cfg['num_reg_fcs'] == len(bbox_head.reg_fcs) // 3
            assert bbox_cfg['in_channels'] == bbox_head.in_channels
            assert bbox_cfg['in_channels'] == bbox_head.fc_cls.in_features
            assert bbox_cfg['in_channels'] == bbox_head.fc_reg.in_features
            assert bbox_cfg['in_channels'] == bbox_head.attention.embed_dims
            assert bbox_cfg[
                'feedforward_channels'] == bbox_head.ffn.feedforward_channels

        else:
            assert bbox_cfg.in_channels == bbox_head.in_channels
            with_cls = bbox_cfg.get('with_cls', True)

            if with_cls:
                fc_out_channels = bbox_cfg.get('fc_out_channels', 2048)
                assert (fc_out_channels == bbox_head.fc_cls.in_features)
                assert (bbox_cfg.num_classes +
                        1 == bbox_head.fc_cls.out_features)
            with_reg = bbox_cfg.get('with_reg', True)
            if with_reg:
                out_dim = (4 if bbox_cfg.reg_class_agnostic else 4 *
                           bbox_cfg.num_classes)
                assert bbox_head.fc_reg.out_features == out_dim


def _check_anchorhead(config, head):
    # check consistency between head_config and roi_head
    assert config['type'] == head.__class__.__name__
    assert config.in_channels == head.in_channels

    num_classes = (
        config.num_classes -
        1 if config.loss_cls.get('use_sigmoid', False) else config.num_classes)
    if config['type'] == 'ATSSHead':
        assert (config.feat_channels == head.atss_cls.in_channels)
        assert (config.feat_channels == head.atss_reg.in_channels)
        assert (config.feat_channels == head.atss_centerness.in_channels)
    elif config['type'] == 'SABLRetinaHead':
        assert (config.feat_channels == head.retina_cls.in_channels)
        assert (config.feat_channels == head.retina_bbox_reg.in_channels)
        assert (config.feat_channels == head.retina_bbox_cls.in_channels)
    else:
        assert (config.in_channels == head.conv_cls.in_channels)
        assert (config.in_channels == head.conv_reg.in_channels)
        assert (head.conv_cls.out_channels == num_classes * head.num_anchors)
        assert head.fc_reg.out_channels == 4 * head.num_anchors


# Only tests a representative subset of configurations
# TODO: test pipelines using Albu, current Albu throw None given empty GT
@pytest.mark.parametrize(
    'config_rpath',
    [
        'wider_face/ssd300_wider_face.py',
        'pascal_voc/ssd300_voc0712.py',
        'pascal_voc/ssd512_voc0712.py',
        # 'albu_example/mask_rcnn_r50_fpn_1x.py',
        'foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py',
        'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py',
        'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain_1x_coco.py',
        'fp16/mask_rcnn_r50_fpn_fp16_1x_coco.py'
    ])
def test_config_data_pipeline(config_rpath):
    """Test whether the data pipeline is valid and can process corner cases.

    CommandLine:
        xdoctest -m tests/test_runtime/
            test_config.py test_config_build_data_pipeline
    """
    from mmcv import Config
    from mmdet.datasets.pipelines import Compose
    import numpy as np

    config_dpath = _get_config_directory()
    print(f'Found config_dpath = {config_dpath}')

    def dummy_masks(h, w, num_obj=3, mode='bitmap'):
        assert mode in ('polygon', 'bitmap')
        if mode == 'bitmap':
            masks = np.random.randint(0, 2, (num_obj, h, w), dtype=np.uint8)
            masks = BitmapMasks(masks, h, w)
        else:
            masks = []
            for i in range(num_obj):
                masks.append([])
                masks[-1].append(
                    np.random.uniform(0, min(h - 1, w - 1), (8 + 4 * i, )))
                masks[-1].append(
                    np.random.uniform(0, min(h - 1, w - 1), (10 + 4 * i, )))
            masks = PolygonMasks(masks, h, w)
        return masks

    config_fpath = join(config_dpath, config_rpath)
    cfg = Config.fromfile(config_fpath)

    # remove loading pipeline
    loading_pipeline = cfg.train_pipeline.pop(0)
    loading_ann_pipeline = cfg.train_pipeline.pop(0)
    cfg.test_pipeline.pop(0)

    train_pipeline = Compose(cfg.train_pipeline)
    test_pipeline = Compose(cfg.test_pipeline)

    print(f'Building data pipeline, config_fpath = {config_fpath}')

    print(f'Test training data pipeline: \n{train_pipeline!r}')
    img = np.random.randint(0, 255, size=(888, 666, 3), dtype=np.uint8)
    if loading_pipeline.get('to_float32', False):
        img = img.astype(np.float32)
    mode = 'bitmap' if loading_ann_pipeline.get('poly2mask',
                                                True) else 'polygon'
    results = dict(
        filename='test_img.png',
        ori_filename='test_img.png',
        img=img,
        img_shape=img.shape,
        ori_shape=img.shape,
        gt_bboxes=np.array([[35.2, 11.7, 39.7, 15.7]], dtype=np.float32),
        gt_labels=np.array([1], dtype=np.int64),
        gt_masks=dummy_masks(img.shape[0], img.shape[1], mode=mode),
    )
    results['img_fields'] = ['img']
    results['bbox_fields'] = ['gt_bboxes']
    results['mask_fields'] = ['gt_masks']
    output_results = train_pipeline(results)
    assert output_results is not None

    print(f'Test testing data pipeline: \n{test_pipeline!r}')
    results = dict(
        filename='test_img.png',
        ori_filename='test_img.png',
        img=img,
        img_shape=img.shape,
        ori_shape=img.shape,
        gt_bboxes=np.array([[35.2, 11.7, 39.7, 15.7]], dtype=np.float32),
        gt_labels=np.array([1], dtype=np.int64),
        gt_masks=dummy_masks(img.shape[0], img.shape[1], mode=mode),
    )
    results['img_fields'] = ['img']
    results['bbox_fields'] = ['gt_bboxes']
    results['mask_fields'] = ['gt_masks']
    output_results = test_pipeline(results)
    assert output_results is not None

    # test empty GT
    print('Test empty GT with training data pipeline: '
          f'\n{train_pipeline!r}')
    results = dict(
        filename='test_img.png',
        ori_filename='test_img.png',
        img=img,
        img_shape=img.shape,
        ori_shape=img.shape,
        gt_bboxes=np.zeros((0, 4), dtype=np.float32),
        gt_labels=np.array([], dtype=np.int64),
        gt_masks=dummy_masks(img.shape[0], img.shape[1], num_obj=0, mode=mode),
    )
    results['img_fields'] = ['img']
    results['bbox_fields'] = ['gt_bboxes']
    results['mask_fields'] = ['gt_masks']
    output_results = train_pipeline(results)
    assert output_results is not None

    print(f'Test empty GT with testing data pipeline: \n{test_pipeline!r}')
    results = dict(
        filename='test_img.png',
        ori_filename='test_img.png',
        img=img,
        img_shape=img.shape,
        ori_shape=img.shape,
        gt_bboxes=np.zeros((0, 4), dtype=np.float32),
        gt_labels=np.array([], dtype=np.int64),
        gt_masks=dummy_masks(img.shape[0], img.shape[1], num_obj=0, mode=mode),
    )
    results['img_fields'] = ['img']
    results['bbox_fields'] = ['gt_bboxes']
    results['mask_fields'] = ['gt_masks']
    output_results = test_pipeline(results)
    assert output_results is not None
