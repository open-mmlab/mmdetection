import os.path as osp
import tempfile
from unittest import TestCase

import numpy as np
import pycocotools.mask as mask_util
import torch
from mmengine.fileio import dump

from mmdet.evaluation import CocoMetric
from mmdet.structures.mask import BitmapMasks


class TestCocoMetric(TestCase):

    def _create_dummy_coco_json(self, json_name):
        dummy_mask = np.zeros((10, 10), order='F', dtype=np.uint8)
        dummy_mask[:5, :5] = 1
        rle_mask = mask_util.encode(dummy_mask)
        rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
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
            'segmentation': rle_mask,
        }

        annotation_2 = {
            'id': 2,
            'image_id': 0,
            'category_id': 0,
            'area': 900,
            'bbox': [100, 120, 30, 30],
            'iscrowd': 0,
            'segmentation': rle_mask,
        }

        annotation_3 = {
            'id': 3,
            'image_id': 0,
            'category_id': 1,
            'area': 1600,
            'bbox': [150, 160, 40, 40],
            'iscrowd': 0,
            'segmentation': rle_mask,
        }

        annotation_4 = {
            'id': 4,
            'image_id': 0,
            'category_id': 0,
            'area': 10000,
            'bbox': [250, 260, 100, 100],
            'iscrowd': 0,
            'segmentation': rle_mask,
        }

        categories = [
            {
                'id': 0,
                'name': 'car',
                'supercategory': 'car',
            },
            {
                'id': 1,
                'name': 'bicycle',
                'supercategory': 'bicycle',
            },
        ]

        fake_json = {
            'images': [image],
            'annotations':
            [annotation_1, annotation_2, annotation_3, annotation_4],
            'categories': categories
        }

        dump(fake_json, json_name)

    def _create_dummy_results(self):
        bboxes = np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                           [150, 160, 190, 200], [250, 260, 350, 360]])
        scores = np.array([100.0, 0.98, 0.96, 0.95])
        labels = np.array([0, 0, 1, 0])
        dummy_mask = np.zeros((4, 10, 10), dtype=np.uint8)
        dummy_mask[:, :5, :5] = 1
        return dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels),
            masks=torch.from_numpy(dummy_mask))

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        with self.assertRaisesRegex(KeyError, 'metric should be one of'):
            CocoMetric(ann_file=fake_json_file, metric='unknown')

    def test_evaluate(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        # test single coco dataset evaluation
        coco_metric = CocoMetric(
            ann_file=fake_json_file,
            classwise=False,
            dataset_meta=dict(classes=['car', 'bicycle']),
            outfile_prefix=f'{self.tmp_dir.name}/test')
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate()
        target = {
            'coco/bbox_mAP(%)': 100.0,
            'coco/bbox_mAP_50(%)': 100.0,
            'coco/bbox_mAP_75(%)': 100.0,
            'coco/bbox_mAP_s(%)': 100.0,
            'coco/bbox_mAP_m(%)': 100.0,
            'coco/bbox_mAP_l(%)': 100.0,
        }
        self.assertDictEqual(eval_results, target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.bbox.json')))

        # test box and segm coco dataset evaluation
        coco_metric = CocoMetric(
            ann_file=fake_json_file,
            metric=['bbox', 'segm'],
            classwise=False,
            dataset_meta=dict(classes=['car', 'bicycle']),
            outfile_prefix=f'{self.tmp_dir.name}/test')
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate()
        target = {
            'coco/bbox_mAP(%)': 100.0,
            'coco/bbox_mAP_50(%)': 100.0,
            'coco/bbox_mAP_75(%)': 100.0,
            'coco/bbox_mAP_s(%)': 100.0,
            'coco/bbox_mAP_m(%)': 100.0,
            'coco/bbox_mAP_l(%)': 100.0,
            'coco/segm_mAP(%)': 100.0,
            'coco/segm_mAP_50(%)': 100.0,
            'coco/segm_mAP_75(%)': 100.0,
            'coco/segm_mAP_s(%)': 100.0,
            'coco/segm_mAP_m(%)': 100.0,
            'coco/segm_mAP_l(%)': 100.0,
        }
        self.assertDictEqual(eval_results, target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.bbox.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.segm.json')))

        # test invalid custom metric_items
        with self.assertRaisesRegex(KeyError,
                                    'metric item "invalid" is not supported'):
            coco_metric = CocoMetric(
                ann_file=fake_json_file,
                dataset_meta=dict(classes=['car', 'bicycle']),
                metric_items=['invalid'])
            coco_metric.process({}, [
                dict(
                    pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))
            ])
            coco_metric.evaluate()

        # test custom metric_items
        coco_metric = CocoMetric(
            ann_file=fake_json_file,
            dataset_meta=dict(classes=['car', 'bicycle']),
            metric_items=['mAP_m'])
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate()
        target = {
            'coco/bbox_mAP_m(%)': 100.0,
        }
        self.assertDictEqual(eval_results, target)

    def test_classwise_evaluate(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        # test single coco dataset evaluation
        coco_metric = CocoMetric(
            ann_file=fake_json_file,
            metric='bbox',
            dataset_meta=dict(classes=['car', 'bicycle']),
            classwise=True)
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate()
        target = {
            'coco/bbox_mAP(%)': 100.0,
            'coco/bbox_mAP_50(%)': 100.0,
            'coco/bbox_mAP_75(%)': 100.0,
            'coco/bbox_mAP_s(%)': 100.0,
            'coco/bbox_mAP_m(%)': 100.0,
            'coco/bbox_mAP_l(%)': 100.0,
            'coco/bbox_car_precision(%)': 100.0,
            'coco/bbox_bicycle_precision(%)': 100.0,
        }
        self.assertDictEqual(eval_results, target)

    def test_manually_set_iou_thrs(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)

        # test single coco dataset evaluation
        coco_metric = CocoMetric(
            ann_file=fake_json_file,
            metric='bbox',
            iou_thrs=[0.3, 0.6],
            dataset_meta=dict(classes=['car', 'bicycle']))
        self.assertTrue(
            np.array_equal(coco_metric.iou_thrs, np.array([0.3, 0.6])))

    # TODO: move to fast recall metric
    # def test_fast_eval_recall(self):
    #     # create dummy data
    #     fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
    #     self._create_dummy_coco_json(fake_json_file)
    #     dummy_pred = self._create_dummy_results()
    #
    #     # test default proposal nums
    #     coco_metric = CocoMetric(
    #         ann_file=fake_json_file, metric='proposal_fast')
    #     coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
    #     coco_metric.process(
    #         {},
    #         [dict(pred_instances=dummy_pred, img_id=0,
    #         ori_shape=(640, 640))])
    #     eval_results = coco_metric.evaluate()
    #     target = {'coco/AR@100': 100.0, 'coco/AR@300': 100.0,
    #     'coco/AR@1000': 1.0}
    #     self.assertDictEqual(eval_results, target)
    #
    #     # test manually set proposal nums
    #     coco_metric = CocoMetric(
    #         ann_file=fake_json_file,
    #         metric='proposal_fast',
    #         proposal_nums=(2, 4))
    #     coco_metric.dataset_meta = dict(classes=['car', 'bicycle'])
    #     coco_metric.process(
    #         {},
    #         [dict(pred_instances=dummy_pred, img_id=0,
    #         ori_shape=(640, 640))])
    #     eval_results = coco_metric.evaluate()
    #     target = {'coco/AR@2': 0.5, 'coco/AR@4': 1.0}
    #     self.assertDictEqual(eval_results, target)

    def test_evaluate_proposal(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        coco_metric = CocoMetric(
            ann_file=fake_json_file,
            metric='proposal',
            dataset_meta=dict(classes=['car', 'bicycle']))
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate()
        target = {
            'coco/AR@1(%)': 25.0,
            'coco/AR@10(%)': 100.0,
            'coco/AR@100(%)': 100.0,
            'coco/AR_s@100(%)': 100.0,
            'coco/AR_m@100(%)': 100.0,
            'coco/AR_l@100(%)': 100.0
        }
        self.assertDictEqual(eval_results, target)

    def test_empty_results(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        coco_metric = CocoMetric(
            ann_file=fake_json_file,
            dataset_meta=dict(classes=['car', 'bicycle']),
            metric='bbox')
        bboxes = np.zeros((0, 4))
        labels = np.array([])
        scores = np.array([])
        dummy_mask = np.zeros((0, 10, 10), dtype=np.uint8)
        empty_pred = dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels),
            masks=torch.from_numpy(dummy_mask))
        coco_metric.process(
            {},
            [dict(pred_instances=empty_pred, img_id=0, ori_shape=(640, 640))])
        # coco api Index error will be caught
        coco_metric.evaluate()

    def test_evaluate_without_json(self):
        dummy_pred = self._create_dummy_results()

        # create fake gts
        bboxes = torch.Tensor([[50, 60, 70, 80], [100, 120, 130, 150],
                               [150, 160, 190, 200], [250, 260, 350, 360]])
        labels = torch.Tensor([0, 0, 1, 0])
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :5] = 1

        dummy_mask = BitmapMasks(
            masks=[mask for _ in range(4)], height=10, width=10)

        dummy_gt = dict(
            img_id=0,
            width=640,
            height=640,
            bboxes=bboxes,
            labels=labels,
            masks=dummy_mask)

        dummy_gt_ignore = dict(
            img_id=0,
            width=640,
            height=640,
            bboxes=torch.zeros((0, 4)),
            labels=torch.zeros((0, )),
            masks=BitmapMasks(masks=[], height=640, width=640))

        # gt area based on bboxes
        coco_metric = CocoMetric(
            ann_file=None,
            metric=['bbox', 'segm'],
            classwise=False,
            gt_mask_area=False,
            dataset_meta=dict(classes=['car', 'bicycle']),
            outfile_prefix=f'{self.tmp_dir.name}/test')
        coco_metric.process({}, [
            dict(
                pred_instances=dummy_pred,
                img_id=0,
                ori_shape=(640, 640),
                gt_instances=dummy_gt,
                ignored_instances=dummy_gt_ignore)
        ])
        eval_results = coco_metric.evaluate()
        target = {
            'coco/bbox_mAP(%)': 100.0,
            'coco/bbox_mAP_50(%)': 100.0,
            'coco/bbox_mAP_75(%)': 100.0,
            'coco/bbox_mAP_s(%)': 100.0,
            'coco/bbox_mAP_m(%)': 100.0,
            'coco/bbox_mAP_l(%)': 100.0,
            'coco/segm_mAP(%)': 100.0,
            'coco/segm_mAP_50(%)': 100.0,
            'coco/segm_mAP_75(%)': 100.0,
            'coco/segm_mAP_s(%)': 100.0,
            'coco/segm_mAP_m(%)': 100.0,
            'coco/segm_mAP_l(%)': 100.0,
        }
        self.assertDictEqual(eval_results, target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.bbox.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.segm.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.gt.json')))

        # gt area based on masks
        coco_metric = CocoMetric(
            ann_file=None,
            metric=['bbox', 'segm'],
            classwise=False,
            gt_mask_area=True,
            dataset_meta=dict(classes=['car', 'bicycle']),
            outfile_prefix=f'{self.tmp_dir.name}/test')
        coco_metric.process({}, [
            dict(
                pred_instances=dummy_pred,
                img_id=0,
                ori_shape=(640, 640),
                gt_instances=dummy_gt,
                ignored_instances=dummy_gt_ignore)
        ])
        eval_results = coco_metric.evaluate()
        target = {
            'coco/bbox_mAP(%)': 100.0,
            'coco/bbox_mAP_50(%)': 100.0,
            'coco/bbox_mAP_75(%)': 100.0,
            'coco/bbox_mAP_s(%)': 100.0,
            'coco/bbox_mAP_m(%)': -100.0,
            'coco/bbox_mAP_l(%)': -100.0,
            'coco/segm_mAP(%)': 100.0,
            'coco/segm_mAP_50(%)': 100.0,
            'coco/segm_mAP_75(%)': 100.0,
            'coco/segm_mAP_s(%)': 100.0,
            'coco/segm_mAP_m(%)': -100.0,
            'coco/segm_mAP_l(%)': -100.0,
        }
        self.assertDictEqual(eval_results, target)

    def test_format_only(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_coco_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        with self.assertRaises(AssertionError):
            CocoMetric(
                ann_file=fake_json_file,
                classwise=False,
                format_only=True,
                outfile_prefix=None)

        coco_metric = CocoMetric(
            ann_file=fake_json_file,
            metric='bbox',
            classwise=False,
            format_only=True,
            dataset_meta=dict(classes=['car', 'bicycle']),
            outfile_prefix=f'{self.tmp_dir.name}/test')
        coco_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = coco_metric.evaluate()
        self.assertDictEqual(eval_results, dict())
        self.assertTrue(osp.exists(f'{self.tmp_dir.name}/test.bbox.json'))
