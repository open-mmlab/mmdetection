# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase, mock
from unittest.mock import Mock, patch

import mmcv
import mmengine
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmengine.utils import is_list_of
from parameterized import parameterized

from mmdet.apis import DetInferencer
from mmdet.evaluation.functional import get_classes
from mmdet.structures import DetDataSample


class TestDetInferencer(TestCase):

    @mock.patch('mmengine.infer.infer._load_checkpoint', return_value=None)
    def test_init(self, mock):
        # init from metafile
        DetInferencer('rtmdet-t')
        # init from cfg
        DetInferencer('configs/yolox/yolox_tiny_8xb8-300e_coco.py')

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
        'rtmdet-t', 'mask-rcnn_r50_fpn_1x_coco', 'panoptic_fpn_r50_fpn_1x_coco'
    ])
    def test_call(self, model):
        # single img
        img_path = 'tests/data/color.jpg'

        mock_load = Mock(return_value=None)
        with patch('mmengine.infer.infer._load_checkpoint', mock_load):
            inferencer = DetInferencer(model)

        # In the case of not loading the pretrained weight, the category
        # defaults to COCO 80, so it needs to be replaced.
        if model == 'panoptic_fpn_r50_fpn_1x_coco':
            inferencer.visualizer.dataset_meta = {
                'classes': get_classes('coco_panoptic'),
                'palette': 'random'
            }

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

        # There is a jitter operation when the mask is drawn,
        # so it cannot be asserted.
        if model == 'rtmdet-t':
            for res_bs1_vis, res_bs3_vis in zip(res_bs1['visualization'],
                                                res_bs3['visualization']):
                self.assertTrue(np.allclose(res_bs1_vis, res_bs3_vis))

    @parameterized.expand([
        'rtmdet-t', 'mask-rcnn_r50_fpn_1x_coco', 'panoptic_fpn_r50_fpn_1x_coco'
    ])
    def test_visualize(self, model):
        img_paths = ['tests/data/color.jpg', 'tests/data/gray.jpg']

        mock_load = Mock(return_value=None)
        with patch('mmengine.infer.infer._load_checkpoint', mock_load):
            inferencer = DetInferencer(model)

        # In the case of not loading the pretrained weight, the category
        # defaults to COCO 80, so it needs to be replaced.
        if model == 'panoptic_fpn_r50_fpn_1x_coco':
            inferencer.visualizer.dataset_meta = {
                'classes': get_classes('coco_panoptic'),
                'palette': 'random'
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            inferencer(img_paths, out_dir=tmp_dir)
            for img_dir in ['color.jpg', 'gray.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, 'vis', img_dir)))

    @parameterized.expand([
        'rtmdet-t', 'mask-rcnn_r50_fpn_1x_coco', 'panoptic_fpn_r50_fpn_1x_coco'
    ])
    def test_postprocess(self, model):
        # return_datasamples
        img_path = 'tests/data/color.jpg'

        mock_load = Mock(return_value=None)
        with patch('mmengine.infer.infer._load_checkpoint', mock_load):
            inferencer = DetInferencer(model)

        # In the case of not loading the pretrained weight, the category
        # defaults to COCO 80, so it needs to be replaced.
        if model == 'panoptic_fpn_r50_fpn_1x_coco':
            inferencer.visualizer.dataset_meta = {
                'classes': get_classes('coco_panoptic'),
                'palette': 'random'
            }

        res = inferencer(img_path, return_datasamples=True)
        self.assertTrue(is_list_of(res['predictions'], DetDataSample))

        with tempfile.TemporaryDirectory() as tmp_dir:
            res = inferencer(img_path, out_dir=tmp_dir, no_save_pred=False)
            dumped_res = mmengine.load(
                osp.join(tmp_dir, 'preds', 'color.json'))
            self.assertEqual(res['predictions'][0], dumped_res)

    @mock.patch('mmengine.infer.infer._load_checkpoint', return_value=None)
    def test_pred2dict(self, mock):
        data_sample = DetDataSample()
        data_sample.pred_instances = InstanceData()

        data_sample.pred_instances.bboxes = np.array([[0, 0, 1, 1]])
        data_sample.pred_instances.labels = np.array([0])
        data_sample.pred_instances.scores = torch.FloatTensor([0.9])
        res = DetInferencer('rtmdet-t').pred2dict(data_sample)
        self.assertListAlmostEqual(res['bboxes'], [[0, 0, 1, 1]])
        self.assertListAlmostEqual(res['labels'], [0])
        self.assertListAlmostEqual(res['scores'], [0.9])

    def assertListAlmostEqual(self, list1, list2, places=7):
        for i in range(len(list1)):
            if isinstance(list1[i], list):
                self.assertListAlmostEqual(list1[i], list2[i], places=places)
            else:
                self.assertAlmostEqual(list1[i], list2[i], places=places)
