# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from copy import deepcopy

import mmcv
import numpy as np
import torch

from mmdet.utils import split_batch


def test_split_batch():
    img_root = osp.join(osp.dirname(__file__), '../data/color.jpg')
    img = mmcv.imread(img_root, 'color')
    h, w, _ = img.shape
    gt_bboxes = np.array([[0.2 * w, 0.2 * h, 0.4 * w, 0.4 * h],
                          [0.6 * w, 0.6 * h, 0.8 * w, 0.8 * h]],
                         dtype=np.float32)
    gt_lables = np.ones(gt_bboxes.shape[0], dtype=np.int64)

    img = torch.tensor(img).permute(2, 0, 1)
    meta = dict()
    meta['filename'] = img_root
    meta['ori_shape'] = img.shape
    meta['img_shape'] = img.shape
    meta['img_norm_cfg'] = {
        'mean': np.array([103.53, 116.28, 123.675], dtype=np.float32),
        'std': np.array([1., 1., 1.], dtype=np.float32),
        'to_rgb': False
    }
    meta['pad_shape'] = img.shape
    # For example, tag include sup, unsup_teacher and unsup_student,
    # in order to distinguish the difference between the three groups of data,
    # the scale_factor of sup is [0.5, 0.5, 0.5, 0.5]
    # the scale_factor of unsup_teacher is [1.0, 1.0, 1.0, 1.0]
    # the scale_factor of unsup_student is [2.0, 2.0, 2.0, 2.0]
    imgs = img.unsqueeze(0).repeat(9, 1, 1, 1)
    img_metas = []
    tags = [
        'sup', 'unsup_teacher', 'unsup_student', 'unsup_teacher',
        'unsup_student', 'unsup_teacher', 'unsup_student', 'unsup_teacher',
        'unsup_student'
    ]
    for tag in tags:
        img_meta = deepcopy(meta)
        if tag == 'sup':
            img_meta['scale_factor'] = [0.5, 0.5, 0.5, 0.5]
            img_meta['tag'] = 'sup'
        elif tag == 'unsup_teacher':
            img_meta['scale_factor'] = [1.0, 1.0, 1.0, 1.0]
            img_meta['tag'] = 'unsup_teacher'
        elif tag == 'unsup_student':
            img_meta['scale_factor'] = [2.0, 2.0, 2.0, 2.0]
            img_meta['tag'] = 'unsup_student'
        else:
            continue
        img_metas.append(img_meta)
    kwargs = dict()
    kwargs['gt_bboxes'] = [torch.tensor(gt_bboxes)] + [torch.zeros(0, 4)] * 8
    kwargs['gt_lables'] = [torch.tensor(gt_lables)] + [torch.zeros(0, )] * 8
    data_groups = split_batch(imgs, img_metas, kwargs)
    assert set(data_groups.keys()) == set(tags)
    assert data_groups['sup']['img'].shape == (1, 3, h, w)
    assert data_groups['unsup_teacher']['img'].shape == (4, 3, h, w)
    assert data_groups['unsup_student']['img'].shape == (4, 3, h, w)
    # the scale_factor of sup is [0.5, 0.5, 0.5, 0.5]
    assert data_groups['sup']['img_metas'][0]['scale_factor'] == [
        0.5, 0.5, 0.5, 0.5
    ]
    # the scale_factor of unsup_teacher is [1.0, 1.0, 1.0, 1.0]
    assert data_groups['unsup_teacher']['img_metas'][0]['scale_factor'] == [
        1.0, 1.0, 1.0, 1.0
    ]
    assert data_groups['unsup_teacher']['img_metas'][1]['scale_factor'] == [
        1.0, 1.0, 1.0, 1.0
    ]
    assert data_groups['unsup_teacher']['img_metas'][2]['scale_factor'] == [
        1.0, 1.0, 1.0, 1.0
    ]
    assert data_groups['unsup_teacher']['img_metas'][3]['scale_factor'] == [
        1.0, 1.0, 1.0, 1.0
    ]
    # the scale_factor of unsup_student is [2.0, 2.0, 2.0, 2.0]
    assert data_groups['unsup_student']['img_metas'][0]['scale_factor'] == [
        2.0, 2.0, 2.0, 2.0
    ]
    assert data_groups['unsup_student']['img_metas'][1]['scale_factor'] == [
        2.0, 2.0, 2.0, 2.0
    ]
    assert data_groups['unsup_student']['img_metas'][2]['scale_factor'] == [
        2.0, 2.0, 2.0, 2.0
    ]
    assert data_groups['unsup_student']['img_metas'][3]['scale_factor'] == [
        2.0, 2.0, 2.0, 2.0
    ]
