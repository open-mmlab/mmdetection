from unittest.mock import MagicMock

import pytest

from mmdet.datasets import DATASETS


@pytest.mark.parametrize('dataset', [
    'CocoDataset', 'VOCDataset', 'CityscapesDataset', 'WIDERFaceDataset',
    'CustomDataset'
])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()

    original_classes = dataset_class.CLASSES

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=('foo', 'bar'),
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ('foo', 'bar')

    # Test setting classes as a list
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=['foo', 'bar'],
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['foo', 'bar']

    # Test default behavior
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=None,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES == original_classes

    # Test sending file path
    import tempfile
    import os.path as osp
    tmp_file = tempfile.TemporaryDirectory()
    file_name = osp.join(tmp_file.name, 'label.txt')
    with open(file_name, 'w') as f:
        f.write('foo\nbar\n')

    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=file_name,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['foo', 'bar']
    tmp_file.cleanup()
