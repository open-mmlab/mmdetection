import csv
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest

from mmdet.datasets import OpenImagesChallengeDataset, OpenImagesDataset


def _create_ids_error_oid_csv(
    label_file,
    fake_csv_file,
):
    label_description = ['/m/000002', 'Football']
    # `newline=''` is used to avoid index error of out of bounds
    # in Windows system
    with open(label_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(label_description)

    header = [
        'ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin',
        'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction',
        'IsInside'
    ]
    annotations = [[
        'color', 'xclick', '/m/000002', '1', '0.022673031', '0.9642005',
        '0.07103825', '0.80054647', '0', '0', '0', '0', '0'
    ],
                   [
                       '000595fe6fee6369', 'xclick', '/m/000000', '1', '0',
                       '1', '0', '1', '0', '0', '1', '0', '0'
                   ]]
    # `newline=''` is used to avoid index error of out of bounds
    # in Windows system
    with open(fake_csv_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(annotations)


def _create_oid_style_ann(label_file, csv_file, label_level_file):
    label_description = [['/m/000000', 'Sports equipment'],
                         ['/m/000001', 'Ball'], ['/m/000002', 'Football'],
                         ['/m/000004', 'Bicycle']]
    with open(label_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(label_description)

    header = [
        'ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin',
        'YMax', 'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction',
        'IsInside'
    ]
    annotations = [
        [
            'color', 'xclick', '/m/000002', 1, 0.0333333, 0.1, 0.0333333, 0.1,
            0, 0, 1, 0, 0
        ],
        [
            'color', 'xclick', '/m/000002', 1, 0.1, 0.166667, 0.1, 0.166667, 0,
            0, 0, 0, 0
        ],
    ]
    # `newline=''` is used to avoid index error of out of bounds
    # in Windows system
    with open(csv_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(annotations)

    header = ['ImageID', 'Source', 'LabelName', 'Confidence']
    annotations = [['color', 'xclick', '/m/000002', '1'],
                   ['color', 'xclick', '/m/000004', '0']]
    # `newline=''` is used to avoid index error of out of bounds
    # in Windows system
    with open(label_level_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(annotations)


def _create_hierarchy_json(hierarchy_name):
    fake_hierarchy = \
        {'LabelName':  '/m/0bl9f',      # entity label
         'Subcategory': [
             {
                 'LabelName': '/m/000000',
                 'Subcategory':
                     [
                         {'LabelName': '/m/000001',
                          'Subcategory':
                              [
                                  {
                                      'LabelName': '/m/000002'
                                  }
                              ]
                          },
                         {
                             'LabelName': '/m/000004'
                         }
                     ]
             }
         ]
         }

    mmcv.dump(fake_hierarchy, hierarchy_name)


def _create_hierarchy_np(hierarchy_name):
    fake_hierarchy = np.array([[0, 1, 0, 0, 0], [0, 1, 1, 0,
                                                 0], [0, 1, 1, 1, 0],
                               [0, 1, 0, 0, 1], [0, 0, 0, 0, 0]])
    with open(hierarchy_name, 'wb') as f:
        np.save(f, fake_hierarchy)


def _create_dummy_results():
    boxes = [
        np.zeros((0, 5)),
        np.zeros((0, 5)),
        np.array([[10, 10, 15, 15, 1.0], [15, 15, 30, 30, 0.98],
                  [10, 10, 25, 25, 0.98], [28, 28, 35, 35, 0.97],
                  [30, 30, 51, 51, 0.96], [100, 110, 120, 130, 0.15]]),
        np.array([[30, 30, 50, 50, 0.51]]),
    ]
    return [boxes]


def _creat_oid_challenge_style_ann(txt_file, label_file, label_level_file):
    bboxes = [
        'validation/color.jpg\n',
        '4 29\n',
        '2\n',
        '1 0.0333333 0.1 0.0333333 0.1 1\n',
        '1 0.1 0.166667 0.1 0.166667 0\n',
    ]
    # `newline=''` is used to avoid index error of out of bounds
    # in Windows system
    with open(txt_file, 'w', newline='') as f:
        f.writelines(bboxes)
        f.close()

    label_description = [['/m/000000', 'Sports equipment', 1],
                         ['/m/000001', 'Ball', 2],
                         ['/m/000002', 'Football', 3],
                         ['/m/000004', 'Bicycle', 4]]
    # `newline=''` is used to avoid index error of out of bounds
    # in Windows system
    with open(label_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(label_description)

    header = ['ImageID', 'LabelName', 'Confidence']
    annotations = [['color', '/m/000001', '1'], ['color', '/m/000000', '0']]
    # `newline=''` is used to avoid index error of out of bounds
    # in Windows system
    with open(label_level_file, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(annotations)


def _create_metas(meta_file):

    fake_meta = [{
        'filename': 'data/OpenImages/OpenImages/validation/color.jpg',
        'ori_shape': (300, 300, 3)
    }]
    mmcv.dump(fake_meta, meta_file)


def test_oid_annotation_ids_unique():
    # create fake ann files
    tmp_dir = tempfile.TemporaryDirectory()
    fake_label_file = osp.join(tmp_dir.name, 'fake_label.csv')
    fake_ann_file = osp.join(tmp_dir.name, 'fake_ann.csv')
    _create_ids_error_oid_csv(fake_label_file, fake_ann_file)

    # test annotation ids not unique error
    with pytest.raises(AssertionError):
        OpenImagesDataset(
            ann_file=fake_ann_file, label_file=fake_label_file, pipeline=[])
    tmp_dir.cleanup()


def test_openimages_dataset():
    # create fake ann files
    tmp_dir = tempfile.TemporaryDirectory()
    label_file = osp.join(tmp_dir.name, 'label_file.csv')
    ann_file = osp.join(tmp_dir.name, 'ann_file.csv')
    label_level_file = osp.join(tmp_dir.name, 'label_level_file.csv')
    _create_oid_style_ann(label_file, ann_file, label_level_file)

    hierarchy_json = osp.join(tmp_dir.name, 'hierarchy.json')
    _create_hierarchy_json(hierarchy_json)

    # test whether hierarchy_file is not None when set
    # get_parent_classes is True
    with pytest.raises(AssertionError):
        OpenImagesDataset(
            ann_file=ann_file,
            label_file=label_file,
            image_level_ann_file=label_level_file,
            pipeline=[])

    dataset = OpenImagesDataset(
        ann_file=ann_file,
        label_file=label_file,
        image_level_ann_file=label_level_file,
        hierarchy_file=hierarchy_json,
        pipeline=[])
    ann = dataset.get_ann_info(0)
    # two legal detection bboxes with `group_of` parameter
    assert ann['bboxes'].shape[0] == ann['labels'].shape[0] == \
           ann['gt_is_group_ofs'].shape[0] == 2

    # test load metas from pipeline
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(128, 128),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]
    dataset = OpenImagesDataset(
        ann_file=ann_file,
        img_prefix='tests/data',
        label_file=label_file,
        image_level_ann_file=label_level_file,
        load_from_file=False,
        hierarchy_file=hierarchy_json,
        pipeline=test_pipeline)
    dataset.prepare_test_img(0)
    assert len(dataset.test_img_metas) == 1
    result = _create_dummy_results()
    dataset.evaluate(result)

    # test get hierarchy for classes
    hierarchy_json = osp.join(tmp_dir.name, 'hierarchy.json')
    _create_hierarchy_json(hierarchy_json)

    # test with hierarchy file wrong suffix
    with pytest.raises(AssertionError):
        fake_path = osp.join(tmp_dir.name, 'hierarchy.csv')
        dataset.get_relation_matrix(fake_path)

    # test load hierarchy file succseefully
    hierarchy = dataset.get_relation_matrix(hierarchy_json)
    hierarchy_gt = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0],
                             [1, 0, 0, 1]])
    assert np.equal(hierarchy, hierarchy_gt).all()

    # test evaluation
    # create fake metas
    meta_file = osp.join(tmp_dir.name, 'meta.pkl')
    _create_metas(meta_file)

    dataset = OpenImagesDataset(
        ann_file=ann_file,
        label_file=label_file,
        image_level_ann_file=label_level_file,
        hierarchy_file=hierarchy_json,
        meta_file=meta_file,
        pipeline=[])
    # test evaluation with using group_of, adding father classes to
    # GT and annotations, and considering image_level_image,
    # In the first label (Sports equipment): tp = [0, 1, 0, 0, 1],
    # fp = [1, 0, 1, 1, 0]
    # In the second label (Ball), tp = [0, 1, 0, 1], fp = [1, 0, 1, 0].
    # In the third label (Football), tp = [0, 1, 0, 1], fp = [1, 0, 1, 0].
    # In the forth label (Bicycle), tp = [0], fp = [1].
    result = _create_dummy_results()
    parsed_results = dataset.evaluate(result)
    assert np.isclose(parsed_results['mAP'], 0.8333, 1e-4)

    dataset = OpenImagesDataset(
        ann_file=ann_file,
        label_file=label_file,
        load_image_level_labels=False,
        image_level_ann_file=label_level_file,
        hierarchy_file=hierarchy_json,
        meta_file=meta_file,
        pipeline=[])

    # test evaluation with using group_of, adding father classes to
    # GT and annotations, and not considering image_level_image,
    # In the first label (Sports equipment): tp = [0, 1, 0, 0, 1],
    # fp = [1, 0, 1, 1, 0]
    # In the second label (Ball), tp = [0, 1, 0, 1], fp = [1, 0, 1, 0].
    # In the third label (Football), tp = [0, 1, 0, 1], fp = [1, 0, 1, 0].
    # In the forth label (Bicycle), tp = [], fp = [].
    result = _create_dummy_results()
    parsed_results = dataset.evaluate(result)
    assert np.isclose(parsed_results['mAP'], 0.8333, 1e-4)
    tmp_dir.cleanup()


def test_openimages_challenge_dataset():
    # create fake ann files
    tmp_dir = tempfile.TemporaryDirectory()
    ann_file = osp.join(tmp_dir.name, 'ann_file.txt')
    label_file = osp.join(tmp_dir.name, 'label_file.csv')
    label_level_file = osp.join(tmp_dir.name, 'label_level_file.csv')
    _creat_oid_challenge_style_ann(ann_file, label_file, label_level_file)

    dataset = OpenImagesChallengeDataset(
        ann_file=ann_file,
        label_file=label_file,
        load_image_level_labels=False,
        get_supercategory=False,
        pipeline=[])
    ann = dataset.get_ann_info(0)

    # two legal detection bboxes with `group_of` parameter
    assert ann['bboxes'].shape[0] == ann['labels'].shape[0] == \
           ann['gt_is_group_ofs'].shape[0] == 2

    dataset.prepare_train_img(0)
    dataset.prepare_test_img(0)

    meta_file = osp.join(tmp_dir.name, 'meta.pkl')
    _create_metas(meta_file)

    result = _create_dummy_results()
    with pytest.raises(AssertionError):
        fake_json = osp.join(tmp_dir.name, 'hierarchy.json')
        dataset = OpenImagesChallengeDataset(
            ann_file=ann_file,
            label_file=label_file,
            image_level_ann_file=label_level_file,
            hierarchy_file=fake_json,
            meta_file=meta_file,
            pipeline=[])
        dataset.evaluate(result)

    hierarchy_file = osp.join(tmp_dir.name, 'hierarchy.np')
    _create_hierarchy_np(hierarchy_file)
    dataset = OpenImagesChallengeDataset(
        ann_file=ann_file,
        label_file=label_file,
        image_level_ann_file=label_level_file,
        hierarchy_file=hierarchy_file,
        meta_file=meta_file,
        pipeline=[])
    dataset.evaluate(result)
    tmp_dir.cleanup()
