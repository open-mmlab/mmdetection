import os.path as osp
import tempfile

import mmcv

from mmdet.datasets import CocoPanoptic


def _create_panoptic_style_json(json_name):
    image = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name.jpg',
    }

    annotation = {
        'segments_info': [
            {
                'id': 1,
                'category_id': 0,
                'area': 400,
                'bbox': [50, 60, 20, 20],
                'iscrowd': 0
            },
            {
                # Different to instance style json, there
                # are duplicate ids in panoptic style json
                'id': 1,
                'image_id': 0,
                'category_id': 0,
                'area': 900,
                'bbox': [100, 120, 30, 30],
                'iscrowd': 0
            },
            {
                'id': 1,
                'category_id': 0,
                'iscrowd': 0,
                'bbox': [1, 189, 612, 285],
                'area': 70036
            }
        ],
        'file_name':
        '000000000009.png',
        'image_id':
        0
    }

    categories = [{
        'id': 0,
        'name': 'car',
        'supercategory': 'car',
        'isthing': 1,
    }]

    fake_json = {
        'images': [image],
        'annotations': [annotation],
        'categories': categories
    }
    mmcv.dump(fake_json, json_name)


def test_load_panoptic_style_json():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    _create_panoptic_style_json(fake_json_file)

    CocoPanoptic(ann_file=fake_json_file, classes=('car', ), pipeline=[])
