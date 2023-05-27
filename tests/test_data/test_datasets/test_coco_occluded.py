import os.path as osp
from tempfile import TemporaryDirectory

import mmcv
import numpy as np

from mmdet.datasets import OccludedSeparatedCocoDataset


def test_occluded_separated_coco_dataset():
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
    res = [([np.array([[50, 60, 70, 80, 0.77]])] * 2, [[rle]] * 2)] * 3

    tempdir = TemporaryDirectory()
    ann_path = osp.join(tempdir.name, 'coco_occluded.pkl')
    mmcv.dump(ann, ann_path)

    dataset = OccludedSeparatedCocoDataset(
        ann_file='tests/data/coco_sample.json',
        occluded_ann=ann_path,
        separated_ann=ann_path,
        pipeline=[],
        test_mode=True)
    eval_res = dataset.evaluate(res)
    assert isinstance(eval_res, dict)
    assert eval_res['occluded_recall'] == 100
    assert eval_res['separated_recall'] == 100
