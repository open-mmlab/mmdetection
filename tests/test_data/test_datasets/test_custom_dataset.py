import os.path as osp
from unittest.mock import MagicMock, patch

import pytest

import mmcv
import tempfile
import numpy as np

from mmdet.datasets import DATASETS


@patch('mmdet.datasets.CocoDataset.load_annotations', MagicMock)
@patch('mmdet.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmdet.datasets.XMLDataset.load_annotations', MagicMock)
@patch('mmdet.datasets.CityscapesDataset.load_annotations', MagicMock)
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
    tmp_dir = tempfile.TemporaryDirectory()
    # create dummy data
    fake_pkl_file = osp.join(tmp_dir.name, 'fake_data.pkl')
    fake_pkl = [{
        'filename': 'fake_name.jpg',
        'width': 640,
        'height': 640,
        'ann': {
            'bboxes':
                np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                          [150, 160, 190, 200], [250, 260, 350, 360]]),
            'labels':
                np.array([0, 0, 0, 0])
        }
    }]
    mmcv.dump(fake_pkl, fake_pkl_file)
    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        ann_file=fake_pkl_file,
        pipeline=[],
        classes=('bus', 'car'),
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ('bus', 'car')
    assert repr(custom_dataset) == "bus: 4\tcar: 0"

    # Test setting classes as a list
    custom_dataset = dataset_class(
        ann_file=fake_pkl_file,
        pipeline=[],
        classes=['bus', 'car'],
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
    assert repr(custom_dataset) == "bus: 4\tcar: 0"

    # Test overriding not a subset
    custom_dataset = dataset_class(
        ann_file=fake_pkl_file,
        pipeline=[],
        classes=['foo'],
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['foo']
    assert repr(custom_dataset) == "foo: 4"

    # Test default behavior
    custom_dataset = dataset_class(
        ann_file=fake_pkl_file,
        pipeline=[],
        classes=None,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')


    if original_classes is None:
        result = "Dataset is empty"
    else:
        instance_count = np.zeros(len(original_classes))
        instance_count [0] = 4
        result = ''
        for cls, count in enumerate(instance_count):
            result += f"{original_classes[cls]}: {count}\t"
            if (cls + 1) % 5 == 0:
                result += '\n'

    assert custom_dataset.CLASSES == original_classes
    assert repr(custom_dataset) == result

    # Test sending file path
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('bus\ncar\n')
    custom_dataset = dataset_class(
        ann_file=fake_pkl_file,
        pipeline=[],
        classes=tmp_file.name,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')
    tmp_file.close()

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
    assert repr(custom_dataset) == "bus: 4\tcar: 0"


