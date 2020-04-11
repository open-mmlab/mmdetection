from unittest.mock import MagicMock

import pytest

from mmdet.datasets import DATASETS


@pytest.mark.parametrize('dataset', [
    'XMLDataset', 'CocoDataset', 'VOCDataset', 'CityscapesDataset',
    'WIDERFaceDataset', 'CustomDataset'
])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()

    original_classes = dataset_class.CLASSES

    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=('foo', 'bar'),
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ('foo', 'bar')
