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
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

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


def test_cascade_rcnn_aug_test():
    aug_result = model_aug_test_template(
        'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py')
    assert len(aug_result) == 80


def test_mask_rcnn_aug_test():
    aug_result = model_aug_test_template(
        'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py')
    assert len(aug_result) == 2
    assert len(aug_result[0]) == 80
    assert len(aug_result[1]) == 80


def test_htc_aug_test():
    aug_result = model_aug_test_template('configs/htc/htc_r50_fpn_1x_coco.py')
    assert len(aug_result) == 2
    assert len(aug_result[0]) == 80
    assert len(aug_result[1]) == 80
