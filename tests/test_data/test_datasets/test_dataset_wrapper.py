# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import math
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np

from mmdet.datasets import (ClassBalancedDataset, ConcatDataset, CustomDataset,
                            MultiImageMixDataset, RepeatDataset)


def test_dataset_wrapper():
    CustomDataset.load_annotations = MagicMock()
    CustomDataset.__getitem__ = MagicMock(side_effect=lambda idx: idx)
    dataset_a = CustomDataset(
        ann_file=MagicMock(), pipeline=[], test_mode=True, img_prefix='')
    len_a = 10
    cat_ids_list_a = [
        np.random.randint(0, 80, num).tolist()
        for num in np.random.randint(1, 20, len_a)
    ]
    dataset_a.data_infos = MagicMock()
    dataset_a.data_infos.__len__.return_value = len_a
    dataset_a.get_cat_ids = MagicMock(
        side_effect=lambda idx: cat_ids_list_a[idx])
    dataset_b = CustomDataset(
        ann_file=MagicMock(), pipeline=[], test_mode=True, img_prefix='')
    len_b = 20
    cat_ids_list_b = [
        np.random.randint(0, 80, num).tolist()
        for num in np.random.randint(1, 20, len_b)
    ]
    dataset_b.data_infos = MagicMock()
    dataset_b.data_infos.__len__.return_value = len_b
    dataset_b.get_cat_ids = MagicMock(
        side_effect=lambda idx: cat_ids_list_b[idx])

    concat_dataset = ConcatDataset([dataset_a, dataset_b])
    assert concat_dataset[5] == 5
    assert concat_dataset[25] == 15
    assert concat_dataset.get_cat_ids(5) == cat_ids_list_a[5]
    assert concat_dataset.get_cat_ids(25) == cat_ids_list_b[15]
    assert len(concat_dataset) == len(dataset_a) + len(dataset_b)

    repeat_dataset = RepeatDataset(dataset_a, 10)
    assert repeat_dataset[5] == 5
    assert repeat_dataset[15] == 5
    assert repeat_dataset[27] == 7
    assert repeat_dataset.get_cat_ids(5) == cat_ids_list_a[5]
    assert repeat_dataset.get_cat_ids(15) == cat_ids_list_a[5]
    assert repeat_dataset.get_cat_ids(27) == cat_ids_list_a[7]
    assert len(repeat_dataset) == 10 * len(dataset_a)

    category_freq = defaultdict(int)
    for cat_ids in cat_ids_list_a:
        cat_ids = set(cat_ids)
        for cat_id in cat_ids:
            category_freq[cat_id] += 1
    for k, v in category_freq.items():
        category_freq[k] = v / len(cat_ids_list_a)

    mean_freq = np.mean(list(category_freq.values()))
    repeat_thr = mean_freq

    category_repeat = {
        cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
        for cat_id, cat_freq in category_freq.items()
    }

    repeat_factors = []
    for cat_ids in cat_ids_list_a:
        cat_ids = set(cat_ids)
        repeat_factor = max({category_repeat[cat_id] for cat_id in cat_ids})
        repeat_factors.append(math.ceil(repeat_factor))
    repeat_factors_cumsum = np.cumsum(repeat_factors)
    repeat_factor_dataset = ClassBalancedDataset(dataset_a, repeat_thr)
    assert len(repeat_factor_dataset) == repeat_factors_cumsum[-1]
    for idx in np.random.randint(0, len(repeat_factor_dataset), 3):
        assert repeat_factor_dataset[idx] == bisect.bisect_right(
            repeat_factors_cumsum, idx)

    img_scale = (60, 60)
    dynamic_scale = (80, 80)
    pipeline = [
        dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
        dict(
            type='RandomAffine',
            scaling_ratio_range=(0.1, 2),
            border=(-img_scale[0] // 2, -img_scale[1] // 2)),
        dict(
            type='MixUp',
            img_scale=img_scale,
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Resize', keep_ratio=True),
        dict(type='Pad', pad_to_square=True, pad_val=114.0),
    ]

    CustomDataset.load_annotations = MagicMock()
    results = []
    for _ in range(2):
        height = np.random.randint(10, 30)
        weight = np.random.randint(10, 30)
        img = np.ones((height, weight, 3))
        gt_bbox = np.concatenate([
            np.random.randint(1, 5, (2, 2)),
            np.random.randint(1, 5, (2, 2)) + 5
        ],
                                 axis=1)
        gt_labels = np.random.randint(0, 80, 2)
        results.append(dict(gt_bboxes=gt_bbox, gt_labels=gt_labels, img=img))

    CustomDataset.__getitem__ = MagicMock(side_effect=lambda idx: results[idx])
    dataset_a = CustomDataset(
        ann_file=MagicMock(), pipeline=[], test_mode=True, img_prefix='')
    len_a = 2
    cat_ids_list_a = [
        np.random.randint(0, 80, num).tolist()
        for num in np.random.randint(1, 20, len_a)
    ]
    dataset_a.data_infos = MagicMock()
    dataset_a.data_infos.__len__.return_value = len_a
    dataset_a.get_cat_ids = MagicMock(
        side_effect=lambda idx: cat_ids_list_a[idx])

    multi_image_mix_dataset = MultiImageMixDataset(dataset_a, pipeline,
                                                   dynamic_scale)
    for idx in range(len_a):
        results_ = multi_image_mix_dataset[idx]
        assert results_['img'].shape == (dynamic_scale[0], dynamic_scale[1], 3)

    # test skip_type_keys
    multi_image_mix_dataset = MultiImageMixDataset(
        dataset_a,
        pipeline,
        dynamic_scale,
        skip_type_keys=('MixUp', 'RandomFlip', 'Resize', 'Pad'))
    for idx in range(len_a):
        results_ = multi_image_mix_dataset[idx]
        assert results_['img'].shape == (img_scale[0], img_scale[1], 3)
