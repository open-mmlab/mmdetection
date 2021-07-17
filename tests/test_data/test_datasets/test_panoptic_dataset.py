import os.path as osp
import tempfile

import mmcv

from mmdet.datasets import CocoPanopticDataset


def _create_panoptic_style_json(json_name):
    image1 = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name1.jpg',
    }

    image2 = {
        'id': 1,
        'width': 640,
        'height': 800,
        'file_name': 'fake_name2.jpg',
    }

    images = [image1, image2]

    annotations = [
        {
            'segments_info': [{
                'id': 1,
                'category_id': 0,
                'area': 400,
                'bbox': [50, 60, 20, 20],
                'iscrowd': 0
            }, {
                'id': 2,
                'category_id': 1,
                'area': 900,
                'bbox': [100, 120, 30, 30],
                'iscrowd': 0
            }, {
                'id': 3,
                'category_id': 2,
                'iscrowd': 0,
                'bbox': [1, 189, 612, 285],
                'area': 70036
            }],
            'file_name':
            'fake_name1.jpg',
            'image_id':
            0
        },
        {
            'segments_info': [
                {
                    # Different to instance style json, there
                    # are duplicate ids in panoptic style json
                    'id': 1,
                    'category_id': 0,
                    'area': 400,
                    'bbox': [50, 60, 20, 20],
                    'iscrowd': 0
                },
                {
                    'id': 4,
                    'category_id': 1,
                    'area': 900,
                    'bbox': [100, 120, 30, 30],
                    'iscrowd': 1
                },
                {
                    'id': 5,
                    'category_id': 2,
                    'iscrowd': 0,
                    'bbox': [100, 200, 200, 300],
                    'area': 66666
                },
                {
                    'id': 6,
                    'category_id': 0,
                    'iscrowd': 0,
                    'bbox': [1, 189, -10, 285],
                    'area': 70036
                }
            ],
            'file_name':
            'fake_name2.jpg',
            'image_id':
            1
        }
    ]

    categories = [{
        'id': 0,
        'name': 'car',
        'supercategory': 'car',
        'isthing': 1
    }, {
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
        'isthing': 1
    }, {
        'id': 2,
        'name': 'wall',
        'supercategory': 'wall',
        'isthing': 0
    }]

    fake_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
    mmcv.dump(fake_json, json_name)

    return fake_json


def test_load_panoptic_style_json():
    tmp_dir = tempfile.TemporaryDirectory()
    fake_json_file = osp.join(tmp_dir.name, 'fake_data.json')
    fake_json = _create_panoptic_style_json(fake_json_file)

    dataset = CocoPanopticDataset(
        ann_file=fake_json_file,
        classes=[cat['name'] for cat in fake_json['categories']],
        pipeline=[])

    ann = dataset.get_ann_info(0)

    # two legal instances
    assert ann['bboxes'].shape[0] == ann['labels'].shape[0] == 2
    # three masks for both foreground and background
    assert len(ann['masks']) == 3

    ann = dataset.get_ann_info(1)

    # one legal instance, one illegal instance,
    # one crowd instance and one background mask
    assert ann['bboxes'].shape[0] == ann['labels'].shape[0] == 1
    assert ann['bboxes_ignore'].shape[0] == 1
    assert len(ann['masks']) == 3
