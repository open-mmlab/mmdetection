import os.path as osp

from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import PIPELINES


def test_default_format_bundle():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../../data'),
        img_info=dict(filename='color.jpg'))
    load = dict(type='LoadImageFromFile')
    load = build_from_cfg(load, PIPELINES)
    bundle = dict(type='DefaultFormatBundle')
    bundle = build_from_cfg(bundle, PIPELINES)
    results = load(results)
    assert 'pad_shape' not in results
    assert 'scale_factor' not in results
    assert 'img_norm_cfg' not in results
    results = bundle(results)
    assert 'pad_shape' in results
    assert 'scale_factor' in results
    assert 'img_norm_cfg' in results
