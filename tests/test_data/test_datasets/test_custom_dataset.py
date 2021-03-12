import os.path as osp
from unittest.mock import MagicMock, patch

import pytest

import mmcv
import tempfile
import numpy as np

from mmdet.datasets import DATASETS


def _create_dummy_coco_json(json_name):
    image = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name.jpg',
    }

    annotation_1 = {
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 400,
        'bbox': [50, 60, 20, 20],
        'iscrowd': 0,
    }

    annotation_2 = {
        'id': 2,
        'image_id': 0,
        'category_id': 0,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
    }

    annotation_3 = {
        'id': 3,
        'image_id': 0,
        'category_id': 0,
        'area': 1600,
        'bbox': [150, 160, 40, 40],
        'iscrowd': 0,
    }

    annotation_4 = {
        'id': 4,
        'image_id': 0,
        'category_id': 0,
        'area': 10000,
        'bbox': [250, 260, 100, 100],
        'iscrowd': 0,
    }

    categories = [{
        'id': 0,
        'name': 'car',
        'supercategory': 'car',
    }]

    fake_json = {
        'images': [image],
        'annotations':
            [annotation_1, annotation_2, annotation_3, annotation_4],
        'categories': categories
    }

    mmcv.dump(fake_json, json_name)


def _create_dummy_custom_pkl(pkl_name):
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
    mmcv.dump(fake_pkl, pkl_name)

@patch('mmdet.datasets.CocoDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.CustomDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.XMLDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.CityscapesDataset._filter_imgs', MagicMock)
@pytest.mark.parametrize('dataset',
                         ['CocoDataset', 'VOCDataset', 'CityscapesDataset', 'CustomDataset'])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)
    if dataset in ['CocoDataset', 'CityscapesDataset']:
        dataset_class.coco = MagicMock()
        dataset_class.cat_ids = MagicMock()

    original_classes = dataset_class.CLASSES

    # create dummy data
    tmp_dir = tempfile.TemporaryDirectory()
    if dataset in ['CocoDataset', 'CityscapesDataset']:
        fake_file = osp.join(tmp_dir.name, 'fake_data.json')
        _create_dummy_coco_json(fake_file)
    elif dataset in ['VOCDataset']:
        fake_file = 'tests/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    else:
        fake_file = osp.join(tmp_dir.name, 'fake_data.pkl')
        _create_dummy_custom_pkl(fake_file)

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        ann_file=fake_file,
        pipeline=[],
        classes=('bus', 'car'),
        test_mode=True,
        img_prefix='./tests/data/VOCdevkit/VOC2007/' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ('bus', 'car')
    assert repr(custom_dataset) == "bus: 0\tcar: 0" if dataset == 'VOCDataset' else "bus: 4\tcar: 0"

    # Test setting classes as a list
    custom_dataset = dataset_class(
        ann_file=fake_file,
        pipeline=[],
        classes=['bus', 'car'],
        test_mode=True,
        img_prefix='./tests/data/VOCdevkit/VOC2007/' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
    assert repr(custom_dataset) == "bus: 0\tcar: 0" if dataset == 'VOCDataset' else "bus: 4\tcar: 0"

    # Test overriding not a subset
    custom_dataset = dataset_class(
        ann_file=fake_file,
        pipeline=[],
        classes=['foo'],
        test_mode=True,
        img_prefix='./tests/data/VOCdevkit/VOC2007/' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['foo']
    assert repr(custom_dataset) == "foo: 0" if dataset == 'VOCDataset' else "foo: 4"

    # Test default behavior
    custom_dataset = dataset_class(
        ann_file=fake_file,
        pipeline=[],
        classes=None,
        test_mode=True,
        img_prefix='./tests/data/VOCdevkit/VOC2007/' if dataset == 'VOCDataset' else '')

    if original_classes is None:
        result = "Dataset is empty"
    else:
        instance_count = np.zeros(len(original_classes)).astype(int)
        if dataset == 'VOCDataset':
            instance_count[11] = 1
            instance_count[14] = 1
        else:
            instance_count[0] = 4
        result = ''
        for cls, count in enumerate(instance_count):
            result += f"{original_classes[cls]}: {count}"
            if cls + 1 != len(original_classes):
                result += '\t'

            if (cls + 1) % 5 == 0:
                result += '\n'

    assert custom_dataset.CLASSES == original_classes
    assert repr(custom_dataset) == result

    # Test sending file path
    name_file = osp.join(tmp_dir.name, 'fake_name.txt')
    with open(name_file, 'w') as f:
        f.write('bus\ncar\n')
    custom_dataset = dataset_class(
        ann_file=fake_file,
        pipeline=[],
        classes=name_file,
        test_mode=True,
        img_prefix='./tests/data/VOCdevkit/VOC2007/' if dataset == 'VOCDataset' else '')


    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
    assert repr(custom_dataset) == "bus: 0\tcar: 0" if dataset == 'VOCDataset' else "bus: 4\tcar: 0"
