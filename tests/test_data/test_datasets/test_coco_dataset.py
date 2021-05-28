import os.path as osp
import tempfile

import mmcv
import pytest

from mmdet.datasets import CocoDataset


def _create_ids_error_coco_json(json_name):
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
        'id': 1,
        'image_id': 0,
        'category_id': 0,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
    }

    categories = [{
        'id': 0,
        'name': 'car',
        'supercategory': 'car',
    }]

    fake_json = {
        'images': [image],
        'annotations': [annotation_1, annotation_2],
        'categories': categories
    }
    mmcv.dump(fake_json, json_name)


def _create_three_class(json_name):
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
        'category_id': 1,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
    }
    annotation_3 = {
        'id': 3,
        'image_id': 0,
        'category_id': 2,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
    }

    categories = [{
        'id': 0,
        'name': 'person'
    }, {
        'id': 1,
        'name': 'bicycle'
    }, {
        'id': 2,
        'name': 'car'
    }]

    fake_json = {
        'images': [image],
        'annotations': [annotation_1, annotation_2, annotation_3],
        'categories': categories
    }
    mmcv.dump(fake_json, json_name)


def test_coco_annotation_ids_unique():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_ids_error_coco_json(fake_json_file)

    # test annotation ids not unique error
    with pytest.raises(AssertionError):
        CocoDataset(ann_file=fake_json_file, classes=('car', ), pipeline=[])


def test_coco_ids_consisitent_with_name():
    names1 = ('person', 'bicycle', 'car')
    names2 = ('bicycle', 'car', 'person')
    tmp_dir = tempfile.TemporaryDirectory()
    three_class_json_file = osp.join(tmp_dir.name, 'three_class.json')
    _create_three_class(three_class_json_file)
    cat_names1 = CocoDataset(
        ann_file=three_class_json_file, classes=names1, pipeline=[])
    cat_names2 = CocoDataset(
        ann_file=three_class_json_file, classes=names2, pipeline=[])
    assert cat_names1.cat_ids == [0, 1, 2]
    assert cat_names2.cat_ids == [1, 2, 0]
