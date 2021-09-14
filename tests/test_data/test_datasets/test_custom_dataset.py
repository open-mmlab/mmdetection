# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
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


class CustomDatasetTests(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data')
        self.dataset_class = DATASETS.get('XMLDataset')

    def test_data_infos__default_db_directories(self):
        """Test correct data read having a Pacal-VOC directory structure."""
        test_dataset_root = os.path.join(self.data_dir, 'VOCdevkit', 'VOC2007')
        custom_ds = self.dataset_class(
            data_root=test_dataset_root,
            ann_file=os.path.join(test_dataset_root, 'ImageSets', 'Main',
                                  'trainval.txt'),
            pipeline=[],
            classes=('person', 'dog'),
            test_mode=True)

        self.assertListEqual([{
            'id': '000001',
            'filename': 'JPEGImages/000001.jpg',
            'width': 353,
            'height': 500
        }], custom_ds.data_infos)

    def test_data_infos__overridden_db_subdirectories(self):
        """Test correct data read having a customized directory structure."""
        test_dataset_root = os.path.join(self.data_dir, 'custom_dataset')
        custom_ds = self.dataset_class(
            data_root=test_dataset_root,
            ann_file=os.path.join(test_dataset_root, 'trainval.txt'),
            pipeline=[],
            classes=('person', 'dog'),
            test_mode=True,
            img_prefix='',
            img_subdir='images',
            ann_subdir='images')

        self.assertListEqual([{
            'id': '000001',
            'filename': 'images/000001.jpg',
            'width': 353,
            'height': 500
        }], custom_ds.data_infos)
