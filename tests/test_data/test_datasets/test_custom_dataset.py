from unittest.mock import MagicMock, patch

import pytest

from mmdet.datasets import DATASETS


@patch('mmdet.datasets.CocoDataset.load_annotations', MagicMock())
@patch('mmdet.datasets.CustomDataset.load_annotations', MagicMock())
@patch('mmdet.datasets.XMLDataset.load_annotations', MagicMock())
@patch('mmdet.datasets.CityscapesDataset.load_annotations', MagicMock())
@patch('mmdet.datasets.CocoDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.CustomDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.XMLDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.CityscapesDataset._filter_imgs', MagicMock)
@pytest.mark.parametrize('dataset',
                         ['CocoDataset', 'VOCDataset', 'CityscapesDataset'])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)
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
    print(custom_dataset)

    # Test setting classes as a list
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=['bus', 'car'],
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
    print(custom_dataset)

    # Test overriding not a subset
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=['foo'],
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['foo']
    print(custom_dataset)

    # Test default behavior
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=None,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES == original_classes
    print(custom_dataset)

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
    print(custom_dataset)
