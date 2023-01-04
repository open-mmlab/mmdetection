# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import mmcv
import mmengine
import numpy as np
from mmengine.utils import is_list_of
from parameterized import parameterized

from mmdet.apis import MMDetInferencer
from mmdet.structures import DetDataSample


class TestMMDetInferencer(TestCase):

    def test_init(self):
        # init from metafile
        MMDetInferencer('yolox-tiny')
        # init from cfg
        MMDetInferencer('configs/yolox/yolox_tiny_8xb8-300e_coco.py')

    def assert_predictions_equal(self, preds1, preds2):
        for pred1, pred2 in zip(preds1, preds2):
            if 'bboxes' in pred1:
                self.assertTrue(
                    np.allclose(pred1['bboxes'], pred2['bboxes'], 0.1))
            if 'scores' in pred1:
                self.assertTrue(
                    np.allclose(pred1['scores'], pred2['scores'], 0.1))
            if 'labels' in pred1:
                self.assertTrue(np.allclose(pred1['labels'], pred2['labels']))
            if 'panoptic_seg_path' in pred1:
                self.assertTrue(
                    pred1['panoptic_seg_path'] == pred2['panoptic_seg_path'])

    @parameterized.expand([
        'yolox-tiny', 'mask-rcnn_r50_fpn_1x_coco',
        'panoptic_fpn_r50_fpn_1x_coco'
    ])
    def test_call(self, model):
        # single img
        img_path = 'tests/data/color.jpg'
        inferencer = MMDetInferencer(model)
        res_path = inferencer(img_path, return_vis=True)
        # ndarray
        img = mmcv.imread(img_path)
        res_ndarray = inferencer(img, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # multiple images
        img_paths = ['tests/data/color.jpg', 'tests/data/gray.jpg']
        res_path = inferencer(img_paths, return_vis=True)
        # list of ndarray
        imgs = [mmcv.imread(p) for p in img_paths]
        res_ndarray = inferencer(imgs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # img dir, test different batch sizes
        img_dir = 'tests/data/VOCdevkit/VOC2007/JPEGImages/'
        res_bs1 = inferencer(img_dir, batch_size=1, return_vis=True)
        res_bs3 = inferencer(img_dir, batch_size=3, return_vis=True)
        self.assert_predictions_equal(res_bs1['predictions'],
                                      res_bs3['predictions'])
        if model == 'yolox-tiny':
            # There is a jitter operation when the mask is drawn,
            # so it cannot be asserted.
            for res_bs1_vis, res_bs3_vis in zip(res_bs1['visualization'],
                                                res_bs3['visualization']):
                self.assertTrue(np.allclose(res_bs1_vis, res_bs3_vis))

    @parameterized.expand([
        'yolox-tiny', 'mask-rcnn_r50_fpn_1x_coco',
        'panoptic_fpn_r50_fpn_1x_coco'
    ])
    def test_visualize(self, model):
        img_paths = ['tests/data/color.jpg', 'tests/data/gray.jpg']
        inferencer = MMDetInferencer(model)
        # img_out_dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            inferencer(img_paths, img_out_dir=tmp_dir)
            for img_dir in ['color.jpg', 'gray.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, img_dir)))

    @parameterized.expand([
        'yolox-tiny', 'mask-rcnn_r50_fpn_1x_coco',
        'panoptic_fpn_r50_fpn_1x_coco'
    ])
    def test_postprocess(self, model):
        # return_datasample
        img_path = 'tests/data/color.jpg'
        inferencer = MMDetInferencer(model)
        res = inferencer(img_path, return_datasamples=True)
        self.assertTrue(is_list_of(res['predictions'], DetDataSample))

        # pred_out_file
        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_out_file = osp.join(tmp_dir, 'tmp.json')
            res = inferencer(
                img_path, print_result=True, pred_out_file=pred_out_file)
            dumped_res = mmengine.load(pred_out_file)
            self.assert_predictions_equal(res['predictions'],
                                          dumped_res['predictions'])
