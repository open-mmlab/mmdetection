# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np

from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET, CocoPanopticDataset

try:
    from panopticapi.utils import id2rgb
except ImportError:
    id2rgb = None


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


def _create_panoptic_gt_annotations(ann_file):
    categories = [{
        'id': 0,
        'name': 'person',
        'supercategory': 'person',
        'isthing': 1
    }, {
        'id': 1,
        'name': 'dog',
        'supercategory': 'dog',
        'isthing': 1
    }, {
        'id': 2,
        'name': 'wall',
        'supercategory': 'wall',
        'isthing': 0
    }]

    images = [{
        'id': 0,
        'width': 80,
        'height': 60,
        'file_name': 'fake_name1.jpg',
    }]

    annotations = [{
        'segments_info': [{
            'id': 1,
            'category_id': 0,
            'area': 400,
            'bbox': [10, 10, 10, 40],
            'iscrowd': 0
        }, {
            'id': 2,
            'category_id': 0,
            'area': 400,
            'bbox': [30, 10, 10, 40],
            'iscrowd': 0
        }, {
            'id': 3,
            'category_id': 1,
            'iscrowd': 0,
            'bbox': [50, 10, 10, 5],
            'area': 50
        }, {
            'id': 4,
            'category_id': 2,
            'iscrowd': 0,
            'bbox': [0, 0, 80, 60],
            'area': 3950
        }],
        'file_name':
        'fake_name1.png',
        'image_id':
        0
    }]

    gt_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # 4 is the id of the background class annotation.
    gt = np.zeros((60, 80), dtype=np.int64) + 4
    gt_bboxes = np.array([[10, 10, 10, 40], [30, 10, 10, 40], [50, 10, 10, 5]],
                         dtype=np.int64)
    for i in range(3):
        x, y, w, h = gt_bboxes[i]
        gt[y:y + h, x:x + w] = i + 1  # id starts from 1

    gt = id2rgb(gt).astype(np.uint8)
    img_path = osp.join(osp.dirname(ann_file), 'fake_name1.png')
    mmcv.imwrite(gt[:, :, ::-1], img_path)

    mmcv.dump(gt_json, ann_file)
    return gt_json


def test_panoptic_evaluation():
    if id2rgb is None:
        return

    # TP for background class, IoU=3576/4324=0.827
    # 2 the category id of the background class
    pred = np.zeros((60, 80), dtype=np.int64) + 2
    pred_bboxes = np.array(
        [
            [11, 11, 10, 40],  # TP IoU=351/449=0.78
            [38, 10, 10, 40],  # FP
            [51, 10, 10, 5]
        ],  # TP IoU=45/55=0.818
        dtype=np.int64)
    pred_labels = np.array([0, 0, 1], dtype=np.int64)
    for i in range(3):
        x, y, w, h = pred_bboxes[i]
        pred[y:y + h, x:x + w] = (i + 1) * INSTANCE_OFFSET + pred_labels[i]

    tmp_dir = tempfile.TemporaryDirectory()
    ann_file = osp.join(tmp_dir.name, 'panoptic.json')
    gt_json = _create_panoptic_gt_annotations(ann_file)

    results = [{'pan_results': pred}]

    dataset = CocoPanopticDataset(
        ann_file=ann_file,
        seg_prefix=tmp_dir.name,
        classes=[cat['name'] for cat in gt_json['categories']],
        pipeline=[])

    # For 'person', sq = 0.78 / 1, rq = 1 / 2( 1 tp + 0.5 * (1 fn + 1 fp))
    # For 'dog', sq = 0.818, rq = 1 / 1
    # For 'wall', sq = 0.827, rq = 1 / 1
    # Here is the results for all classes:
    # +--------+--------+--------+---------+------------+
    # |        | PQ     | SQ     | RQ      | categories |
    # +--------+--------+--------+---------+------------+
    # | All    | 67.869 | 80.898 | 83.333  |      3     |
    # | Things | 60.453 | 79.996 | 75.000  |      2     |
    # | Stuff  | 82.701 | 82.701 | 100.000 |      1     |
    # +--------+--------+--------+---------+------------+
    parsed_results = dataset.evaluate(results)
    assert np.isclose(parsed_results['PQ'], 67.869)
    assert np.isclose(parsed_results['SQ'], 80.898)
    assert np.isclose(parsed_results['RQ'], 83.333)
    assert np.isclose(parsed_results['PQ_th'], 60.453)
    assert np.isclose(parsed_results['SQ_th'], 79.996)
    assert np.isclose(parsed_results['RQ_th'], 75.000)
    assert np.isclose(parsed_results['PQ_st'], 82.701)
    assert np.isclose(parsed_results['SQ_st'], 82.701)
    assert np.isclose(parsed_results['RQ_st'], 100.000)

    # test jsonfile_prefix
    outfile_prefix = osp.join(tmp_dir.name, 'results')
    parsed_results = dataset.evaluate(results, jsonfile_prefix=outfile_prefix)
    assert np.isclose(parsed_results['PQ'], 67.869)
    assert np.isclose(parsed_results['SQ'], 80.898)
    assert np.isclose(parsed_results['RQ'], 83.333)
    assert np.isclose(parsed_results['PQ_th'], 60.453)
    assert np.isclose(parsed_results['SQ_th'], 79.996)
    assert np.isclose(parsed_results['RQ_th'], 75.000)
    assert np.isclose(parsed_results['PQ_st'], 82.701)
    assert np.isclose(parsed_results['SQ_st'], 82.701)
    assert np.isclose(parsed_results['RQ_st'], 100.000)

    # test classwise
    parsed_results = dataset.evaluate(results, classwise=True)
    assert np.isclose(parsed_results['PQ'], 67.869)
    assert np.isclose(parsed_results['SQ'], 80.898)
    assert np.isclose(parsed_results['RQ'], 83.333)
    assert np.isclose(parsed_results['PQ_th'], 60.453)
    assert np.isclose(parsed_results['SQ_th'], 79.996)
    assert np.isclose(parsed_results['RQ_th'], 75.000)
    assert np.isclose(parsed_results['PQ_st'], 82.701)
    assert np.isclose(parsed_results['SQ_st'], 82.701)
    assert np.isclose(parsed_results['RQ_st'], 100.000)
