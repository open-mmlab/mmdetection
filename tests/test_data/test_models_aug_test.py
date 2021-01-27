import os.path as osp

import mmcv
import torch
from mmcv.parallel import collate
from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import PIPELINES
from mmdet.models import build_detector


def model_aug_test_template(cfg_file):
    # get config
    cfg = mmcv.Config.fromfile(cfg_file)
    # init model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)

    # init test pipeline and set aug test
    load_cfg, multi_scale_cfg = cfg.test_pipeline
    multi_scale_cfg['flip'] = True
    multi_scale_cfg['img_scale'] = [(1333, 800), (800, 600), (640, 480)]

    load = build_from_cfg(load_cfg, PIPELINES)
    transform = build_from_cfg(multi_scale_cfg, PIPELINES)

    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../data'),
        img_info=dict(filename='color.jpg'))
    results = transform(load(results))
    assert len(results['img']) == 6
    assert len(results['img_metas']) == 6

    results['img'] = [collate([x]) for x in results['img']]
    results['img_metas'] = [collate([x]).data[0] for x in results['img_metas']]
    # aug test the model
    model.eval()
    with torch.no_grad():
        aug_result = model(return_loss=False, rescale=True, **results)
    return aug_result


def test_aug_test_size():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../data'),
        img_info=dict(filename='color.jpg'))

    # Define simple pipeline
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)

    # get config
    transform = dict(
        type='MultiScaleFlipAug',
        transforms=[],
        img_scale=[(1333, 800), (800, 600), (640, 480)],
        flip=True,
        flip_direction=['horizontal', 'vertical'])
    multi_aug_test_module = build_from_cfg(transform, PIPELINES)

    results = load(results)
    results = multi_aug_test_module(load(results))
    # len(["original", "horizontal", "vertical"]) *
    # len([(1333, 800), (800, 600), (640, 480)])
    assert len(results['img']) == 9


def test_cascade_rcnn_aug_test():
    aug_result = model_aug_test_template(
        'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py')
    assert len(aug_result[0]) == 80


def test_mask_rcnn_aug_test():
    aug_result = model_aug_test_template(
        'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py')
    assert len(aug_result[0]) == 2
    assert len(aug_result[0][0]) == 80
    assert len(aug_result[0][1]) == 80


def test_htc_aug_test():
    aug_result = model_aug_test_template('configs/htc/htc_r50_fpn_1x_coco.py')
    assert len(aug_result[0]) == 2
    assert len(aug_result[0][0]) == 80
    assert len(aug_result[0][1]) == 80


def test_cornernet_aug_test():
    # get config
    cfg = mmcv.Config.fromfile(
        'configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py')
    # init model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model)

    # init test pipeline and set aug test
    load_cfg, multi_scale_cfg = cfg.test_pipeline
    multi_scale_cfg['flip'] = True
    multi_scale_cfg['scale_factor'] = [0.5, 1.0, 2.0]

    load = build_from_cfg(load_cfg, PIPELINES)
    transform = build_from_cfg(multi_scale_cfg, PIPELINES)

    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../data'),
        img_info=dict(filename='color.jpg'))
    results = transform(load(results))
    assert len(results['img']) == 6
    assert len(results['img_metas']) == 6

    results['img'] = [collate([x]) for x in results['img']]
    results['img_metas'] = [collate([x]).data[0] for x in results['img_metas']]
    # aug test the model
    model.eval()
    with torch.no_grad():
        aug_result = model(return_loss=False, rescale=True, **results)
    assert len(aug_result[0]) == 80
