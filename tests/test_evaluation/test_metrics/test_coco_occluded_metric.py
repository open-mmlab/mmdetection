# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import TemporaryDirectory

import mmengine
import numpy as np

from mmdet.datasets import CocoDataset
from mmdet.evaluation import CocoOccludedSeparatedMetric


def test_coco_occluded_separated_metric():
    ann = [[
        'fake1.jpg', 'person', 8, [219.9, 176.12, 11.14, 34.23], {
            'size': [480, 640],
            'counts': b'nYW31n>2N2FNbA48Kf=?XBDe=m0OM3M4YOPB8_>L4JXao5'
        }
    ]] * 3
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[:5, :5] = 1
    rle = {
        'size': [480, 640],
        'counts': b'nYW31n>2N2FNbA48Kf=?XBDe=m0OM3M4YOPB8_>L4JXao5'
    }
    res = [(None,
            dict(
                img_id=0,
                bboxes=np.array([[50, 60, 70, 80]] * 2),
                masks=[rle] * 2,
                labels=np.array([0, 1], dtype=np.int64),
                scores=np.array([0.77, 0.77])))] * 3

    tempdir = TemporaryDirectory()
    ann_path = osp.join(tempdir.name, 'coco_occluded.pkl')
    mmengine.dump(ann, ann_path)

    metric = CocoOccludedSeparatedMetric(
        ann_file='tests/data/coco_sample.json',
        occluded_ann=ann_path,
        separated_ann=ann_path,
        metric=[])
    metric.dataset_meta = CocoDataset.METAINFO
    eval_res = metric.compute_metrics(res)
    assert isinstance(eval_res, dict)
    assert eval_res['occluded_recall'] == 100
    assert eval_res['separated_recall'] == 100
