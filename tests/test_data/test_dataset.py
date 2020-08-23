import bisect
import logging
import math
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.runner import EpochBasedRunner
from torch.utils.data import DataLoader

from mmdet.core.evaluation import DistEvalHook, EvalHook
from mmdet.datasets import (DATASETS, ClassBalancedDataset, ConcatDataset,
                            CustomDataset, RepeatDataset)


@pytest.mark.parametrize('dataset',
                         ['CocoDataset', 'VOCDataset', 'CityscapesDataset'])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    if dataset in ['CocoDataset', 'CityscapesDataset']:
        dataset_class.coco = MagicMock()
        dataset_class.cat_ids = MagicMock()

    original_classes = dataset_class.CLASSES

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=('bus', 'car'),
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ('bus', 'car')
    assert custom_dataset.custom_classes

    # Test setting classes as a list
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=['bus', 'car'],
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
    assert custom_dataset.custom_classes

    # Test overriding not a subset
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=['foo'],
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['foo']
    assert custom_dataset.custom_classes

    # Test default behavior
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=None,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES == original_classes
    assert not custom_dataset.custom_classes

    # Test sending file path
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('bus\ncar\n')
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=tmp_file.name,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')
    tmp_file.close()

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
    assert custom_dataset.custom_classes


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


@patch('mmdet.apis.single_gpu_test', MagicMock)
@patch('mmdet.apis.multi_gpu_test', MagicMock)
@pytest.mark.parametrize('EvalHookParam', (EvalHook, DistEvalHook))
def test_evaluation_hook(EvalHookParam):
    # create dummy data
    dataloader = DataLoader(torch.ones((5, 2)))

    # 0.1. dataloader is not a DataLoader object
    with pytest.raises(TypeError):
        EvalHookParam(dataloader=MagicMock(), interval=-1)

    # 0.2. negative interval
    with pytest.raises(ValueError):
        EvalHookParam(dataloader, interval=-1)

    # 1. start=None, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 2. start=1, interval=1: perform evaluation after each epoch.
    runner = _build_demo_runner()

    evalhook = EvalHookParam(dataloader, start=1, interval=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 2

    # 3. start=None, interval=2: perform evaluation after epoch 2, 4, 6, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 1  # after epoch 2

    # 4. start=1, interval=2: perform evaluation after epoch 1, 3, 5, etc
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=1, interval=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 1 & 3

    # 5. start=0/-1, interval=1: perform evaluation after each epoch and before
    #    epoch 1.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=-1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner.run([dataloader], [('train', 1)], 2)
    assert evalhook.evaluate.call_count == 3  # before epoch1 and after e1 & e2

    # 6. resuming from epoch i, start = x (x<=i), interval =1: perform
    #    evaluation after each epoch and before the first epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=1)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 2
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # before & after epoch 3

    # 7. resuming from epoch i, start = i+1/None, interval =1: perform
    #    evaluation after each epoch.
    runner = _build_demo_runner()
    evalhook = EvalHookParam(dataloader, start=2)
    evalhook.evaluate = MagicMock()
    runner.register_hook(evalhook)
    runner._epoch = 1
    runner.run([dataloader], [('train', 1)], 3)
    assert evalhook.evaluate.call_count == 2  # after epoch 2 & 3


def _build_demo_runner():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()
    tmp_dir = tempfile.mkdtemp()

    runner = EpochBasedRunner(
        model=model, work_dir=tmp_dir, logger=logging.getLogger())
    return runner
