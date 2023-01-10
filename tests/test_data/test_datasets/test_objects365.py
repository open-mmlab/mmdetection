# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import pytest

from mmdet.datasets import Objects365V1Dataset, Objects365V2Dataset


def _create_objects365_json(json_name):
    images = [{
        'file_name': 'fake1.jpg',
        'height': 800,
        'width': 800,
        'id': 0
    }, {
        'file_name': 'fake2.jpg',
        'height': 800,
        'width': 800,
        'id': 1
    }, {
        'file_name': 'patch16/objects365_v2_00908726.jpg',
        'height': 800,
        'width': 800,
        'id': 2
    }]

    annotations = [{
        'bbox': [0, 0, 20, 20],
        'area': 400.00,
        'score': 1.0,
        'category_id': 1,
        'id': 1,
        'image_id': 0
    }, {
        'bbox': [0, 0, 20, 20],
        'area': 400.00,
        'score': 1.0,
        'category_id': 2,
        'id': 2,
        'image_id': 0
    }, {
        'bbox': [0, 0, 20, 20],
        'area': 400.00,
        'score': 1.0,
        'category_id': 1,
        'id': 3,
        'image_id': 1
    }, {
        'bbox': [0, 0, 20, 20],
        'area': 400.00,
        'score': 1.0,
        'category_id': 1,
        'id': 4,
        'image_id': 2
    }]

    categories = [{
        'id': 1,
        'name': 'bus',
        'supercategory': 'none'
    }, {
        'id': 2,
        'name': 'car',
        'supercategory': 'none'
    }]

    fake_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    print(fake_json)
    mmcv.dump(fake_json, json_name)


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


@pytest.mark.parametrize('datasets',
                         [Objects365V1Dataset, Objects365V2Dataset])
def test_annotation_ids_unique(datasets):
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_ids_error_coco_json(fake_json_file)

    # test annotation ids not unique error
    with pytest.raises(AssertionError):
        datasets(ann_file=fake_json_file, classes=('car', ), pipeline=[])

    tmp_dir.cleanup()


def test_load_objects365v1_annotations():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_objects365_json(fake_json_file)

    dataset = Objects365V1Dataset(
        ann_file=fake_json_file, classes=('bus', 'car'), pipeline=[])

    # The Objects365V1Dataset do not filter the `objv2_ignore_list`
    assert len(dataset.data_infos) == 3
    tmp_dir.cleanup()


def test_load_objects365v2_annotations():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_objects365_json(fake_json_file)

    dataset = Objects365V2Dataset(
        ann_file=fake_json_file, classes=('bus', 'car'), pipeline=[])

    # The Objects365V2Dataset need filter the `objv2_ignore_list`
    assert len(dataset.data_infos) == 2
    tmp_dir.cleanup()
