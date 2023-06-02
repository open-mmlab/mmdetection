# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

import mmcv
import numpy as np

from mmdet.core import encode_mask_results
from mmdet.datasets.api_wrappers import pq_compute_single_core
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

    # test the api wrapper of `pq_compute_single_core`
    # Codes are copied from `coco_panoptic.py` and modified
    result_files, _ = dataset.format_results(
        results, jsonfile_prefix=outfile_prefix)

    imgs = dataset.coco.imgs
    gt_json = dataset.coco.img_ann_map  # image to annotations
    gt_json = [{
        'image_id': k,
        'segments_info': v,
        'file_name': imgs[k]['segm_file']
    } for k, v in gt_json.items()]
    pred_json = mmcv.load(result_files['panoptic'])
    pred_json = dict((el['image_id'], el) for el in pred_json['annotations'])

    # match the gt_anns and pred_anns in the same image
    matched_annotations_list = []
    for gt_ann in gt_json:
        img_id = gt_ann['image_id']
        matched_annotations_list.append((gt_ann, pred_json[img_id]))
    gt_folder = dataset.seg_prefix
    pred_folder = osp.join(osp.dirname(outfile_prefix), 'panoptic')

    pq_stat = pq_compute_single_core(0, matched_annotations_list, gt_folder,
                                     pred_folder, dataset.categories)
    pq_all = pq_stat.pq_average(dataset.categories, isthing=None)[0]
    assert np.isclose(pq_all['pq'] * 100, 67.869)
    assert np.isclose(pq_all['sq'] * 100, 80.898)
    assert np.isclose(pq_all['rq'] * 100, 83.333)
    assert pq_all['n'] == 3


def _create_instance_segmentation_gt_annotations(ann_file):
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

    person1_polygon = [10, 10, 20, 10, 20, 50, 10, 50, 10, 10]
    person2_polygon = [30, 10, 40, 10, 40, 50, 30, 50, 30, 10]
    dog_polygon = [50, 10, 60, 10, 60, 15, 50, 15, 50, 10]

    annotations = [
        {
            'id': 0,
            'image_id': 0,
            'category_id': 0,
            'segmentation': [person1_polygon],
            'area': 400,
            'bbox': [10, 10, 10, 40],
            'iscrowd': 0
        },
        {
            'id': 1,
            'image_id': 0,
            'category_id': 0,
            'segmentation': [person2_polygon],
            'area': 400,
            'bbox': [30, 10, 10, 40],
            'iscrowd': 0
        },
        {
            'id': 2,
            'image_id': 0,
            'category_id': 1,
            'segmentation': [dog_polygon],
            'area': 50,
            'bbox': [50, 10, 10, 5],
            'iscrowd': 0
        },
    ]

    gt_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    mmcv.dump(gt_json, ann_file)


def test_instance_segmentation_evaluation():
    pred_bbox = [
        np.array([[11, 10, 20, 50, 0.8], [31, 10, 40, 50, 0.8]]),
        np.array([[51, 10, 60, 15, 0.7]])
    ]

    person1_mask = np.zeros((60, 80), dtype=bool)
    person1_mask[20:50, 11:20] = True
    person2_mask = np.zeros((60, 80), dtype=bool)
    person2_mask[20:50, 31:40] = True
    dog_mask = np.zeros((60, 80), dtype=bool)
    dog_mask[10:15, 51:60] = True

    pred_mask = [[person1_mask, person2_mask], [
        dog_mask,
    ]]
    results = [{'ins_results': (pred_bbox, encode_mask_results(pred_mask))}]

    tmp_dir = tempfile.TemporaryDirectory()
    pan_ann_file = osp.join(tmp_dir.name, 'panoptic.json')
    ins_ann_file = osp.join(tmp_dir.name, 'instance.json')
    _create_panoptic_gt_annotations(pan_ann_file)
    _create_instance_segmentation_gt_annotations(ins_ann_file)

    dataset = CocoPanopticDataset(
        ann_file=pan_ann_file,
        ins_ann_file=ins_ann_file,
        seg_prefix=tmp_dir.name,
        pipeline=[])
    dataset.THING_CLASSES = ['person', 'dog']
    dataset.STUFF_CLASSES = ['wall']
    dataset.CLASSES = dataset.THING_CLASSES + dataset.STUFF_CLASSES
    parsed_results = dataset.evaluate(results, metric=['segm', 'bbox'])

    # Here is the results for instance segmentation:
    # {
    #     'segm_mAP': 0.5005, 'segm_mAP_50': 0.626, 'segm_mAP_75': 0.5,
    #     'segm_mAP_s': 0.5, 'segm_mAP_m': -1.0, 'segm_mAP_l': -1.0,
    #     'segm_mAP_copypaste': '0.500 0.626 0.500 0.500 -1.000 -1.000',
    #     'bbox_mAP': 0.5636, 'bbox_mAP_50': 0.626, 'bbox_mAP_75': 0.626,
    #     'bbox_mAP_s': 0.564, 'bbox_mAP_m': -1.0, 'bbox_mAP_l': -1.0,
    #     'bbox_mAP_copypaste': '0.564 0.626 0.626 0.564 -1.000 -1.000'
    # }

    assert np.isclose(parsed_results['segm_mAP'], 0.5005)
    assert np.isclose(parsed_results['bbox_mAP'], 0.5636)
