import os.path as osp
import tempfile
import unittest

import numpy as np
import pycocotools.mask as mask_util
import torch

from mmdet.evaluation.metrics import LVISMetric

try:
    import lvis
except ImportError:
    lvis = None

from mmengine.fileio import dump


class TestLVISMetric(unittest.TestCase):

    def _create_dummy_lvis_json(self, json_name):
        dummy_mask = np.zeros((10, 10), order='F', dtype=np.uint8)
        dummy_mask[:5, :5] = 1
        rle_mask = mask_util.encode(dummy_mask)
        rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
        image = {
            'id': 0,
            'width': 640,
            'height': 640,
            'neg_category_ids': [],
            'not_exhaustive_category_ids': [],
            'coco_url': 'http://images.cocodataset.org/val2017/0.jpg',
        }

        annotation_1 = {
            'id': 1,
            'image_id': 0,
            'category_id': 1,
            'area': 400,
            'bbox': [50, 60, 20, 20],
            'segmentation': rle_mask,
        }

        annotation_2 = {
            'id': 2,
            'image_id': 0,
            'category_id': 1,
            'area': 900,
            'bbox': [100, 120, 30, 30],
            'segmentation': rle_mask,
        }

        annotation_3 = {
            'id': 3,
            'image_id': 0,
            'category_id': 2,
            'area': 1600,
            'bbox': [150, 160, 40, 40],
            'segmentation': rle_mask,
        }

        annotation_4 = {
            'id': 4,
            'image_id': 0,
            'category_id': 1,
            'area': 10000,
            'bbox': [250, 260, 100, 100],
            'segmentation': rle_mask,
        }

        categories = [
            {
                'id': 1,
                'name': 'aerosol_can',
                'frequency': 'c',
                'image_count': 64
            },
            {
                'id': 2,
                'name': 'air_conditioner',
                'frequency': 'f',
                'image_count': 364
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
        scores = np.array([1.0, 0.98, 0.96, 0.95])
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

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_init(self):
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_lvis_json(fake_json_file)
        with self.assertRaisesRegex(KeyError, 'metric should be one of'):
            LVISMetric(ann_file=fake_json_file, metric='unknown')

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_evaluate(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_lvis_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        # test single lvis dataset evaluation
        lvis_metric = LVISMetric(
            ann_file=fake_json_file,
            classwise=False,
            outfile_prefix=f'{self.tmp_dir.name}/test')
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        lvis_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = lvis_metric.evaluate(size=1)
        target = {
            'lvis/bbox_AP': 1.0,
            'lvis/bbox_AP50': 1.0,
            'lvis/bbox_AP75': 1.0,
            'lvis/bbox_APs': 1.0,
            'lvis/bbox_APm': 1.0,
            'lvis/bbox_APl': 1.0,
            'lvis/bbox_APr': -1.0,
            'lvis/bbox_APc': 1.0,
            'lvis/bbox_APf': 1.0
        }
        self.assertDictEqual(eval_results, target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.bbox.json')))

        # test box and segm lvis dataset evaluation
        lvis_metric = LVISMetric(
            ann_file=fake_json_file,
            metric=['bbox', 'segm'],
            classwise=False,
            outfile_prefix=f'{self.tmp_dir.name}/test')
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        lvis_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = lvis_metric.evaluate(size=1)
        target = {
            'lvis/bbox_AP': 1.0,
            'lvis/bbox_AP50': 1.0,
            'lvis/bbox_AP75': 1.0,
            'lvis/bbox_APs': 1.0,
            'lvis/bbox_APm': 1.0,
            'lvis/bbox_APl': 1.0,
            'lvis/bbox_APr': -1.0,
            'lvis/bbox_APc': 1.0,
            'lvis/bbox_APf': 1.0,
            'lvis/segm_AP': 1.0,
            'lvis/segm_AP50': 1.0,
            'lvis/segm_AP75': 1.0,
            'lvis/segm_APs': 1.0,
            'lvis/segm_APm': 1.0,
            'lvis/segm_APl': 1.0,
            'lvis/segm_APr': -1.0,
            'lvis/segm_APc': 1.0,
            'lvis/segm_APf': 1.0
        }
        self.assertDictEqual(eval_results, target)
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.bbox.json')))
        self.assertTrue(
            osp.isfile(osp.join(self.tmp_dir.name, 'test.segm.json')))

        # test invalid custom metric_items
        with self.assertRaisesRegex(
                KeyError,
                "metric should be one of 'bbox', 'segm', 'proposal', "
                "'proposal_fast', but got invalid."):
            lvis_metric = LVISMetric(
                ann_file=fake_json_file, metric=['invalid'])
            lvis_metric.evaluate(size=1)

        # test custom metric_items
        lvis_metric = LVISMetric(ann_file=fake_json_file, metric_items=['APm'])
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        lvis_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = lvis_metric.evaluate(size=1)
        target = {
            'lvis/bbox_APm': 1.0,
        }
        self.assertDictEqual(eval_results, target)

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_classwise_evaluate(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_lvis_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        # test single lvis dataset evaluation
        lvis_metric = LVISMetric(
            ann_file=fake_json_file, metric='bbox', classwise=True)
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        lvis_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = lvis_metric.evaluate(size=1)
        target = {
            'lvis/bbox_AP': 1.0,
            'lvis/bbox_AP50': 1.0,
            'lvis/bbox_AP75': 1.0,
            'lvis/bbox_APs': 1.0,
            'lvis/bbox_APm': 1.0,
            'lvis/bbox_APl': 1.0,
            'lvis/bbox_APr': -1.0,
            'lvis/bbox_APc': 1.0,
            'lvis/bbox_APf': 1.0,
            'lvis/aerosol_can_precision': 1.0,
            'lvis/air_conditioner_precision': 1.0,
        }
        self.assertDictEqual(eval_results, target)

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_manually_set_iou_thrs(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_lvis_json(fake_json_file)

        # test single lvis dataset evaluation
        lvis_metric = LVISMetric(
            ann_file=fake_json_file, metric='bbox', iou_thrs=[0.3, 0.6])
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        self.assertEqual(lvis_metric.iou_thrs, [0.3, 0.6])

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_fast_eval_recall(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_lvis_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        # test default proposal nums
        lvis_metric = LVISMetric(
            ann_file=fake_json_file, metric='proposal_fast')
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        lvis_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = lvis_metric.evaluate(size=1)
        target = {'lvis/AR@100': 1.0, 'lvis/AR@300': 1.0, 'lvis/AR@1000': 1.0}
        self.assertDictEqual(eval_results, target)

        # test manually set proposal nums
        lvis_metric = LVISMetric(
            ann_file=fake_json_file,
            metric='proposal_fast',
            proposal_nums=(2, 4))
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        lvis_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = lvis_metric.evaluate(size=1)
        target = {'lvis/AR@2': 0.5, 'lvis/AR@4': 1.0}
        self.assertDictEqual(eval_results, target)

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_evaluate_proposal(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_lvis_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        lvis_metric = LVISMetric(ann_file=fake_json_file, metric='proposal')
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        lvis_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = lvis_metric.evaluate(size=1)
        target = {
            'lvis/AR@300': 1.0,
            'lvis/ARs@300': 1.0,
            'lvis/ARm@300': 1.0,
            'lvis/ARl@300': 1.0
        }
        self.assertDictEqual(eval_results, target)

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_empty_results(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_lvis_json(fake_json_file)
        lvis_metric = LVISMetric(ann_file=fake_json_file, metric='bbox')
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        bboxes = np.zeros((0, 4))
        labels = np.array([])
        scores = np.array([])
        dummy_mask = np.zeros((0, 10, 10), dtype=np.uint8)
        empty_pred = dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels),
            masks=torch.from_numpy(dummy_mask))
        lvis_metric.process(
            {},
            [dict(pred_instances=empty_pred, img_id=0, ori_shape=(640, 640))])
        # lvis api Index error will be caught
        lvis_metric.evaluate(size=1)

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_format_only(self):
        # create dummy data
        fake_json_file = osp.join(self.tmp_dir.name, 'fake_data.json')
        self._create_dummy_lvis_json(fake_json_file)
        dummy_pred = self._create_dummy_results()

        with self.assertRaises(AssertionError):
            LVISMetric(
                ann_file=fake_json_file,
                classwise=False,
                format_only=True,
                outfile_prefix=None)

        lvis_metric = LVISMetric(
            ann_file=fake_json_file,
            metric='bbox',
            classwise=False,
            format_only=True,
            outfile_prefix=f'{self.tmp_dir.name}/test')
        lvis_metric.dataset_meta = dict(
            classes=['aerosol_can', 'air_conditioner'])
        lvis_metric.process(
            {},
            [dict(pred_instances=dummy_pred, img_id=0, ori_shape=(640, 640))])
        eval_results = lvis_metric.evaluate(size=1)
        self.assertDictEqual(eval_results, dict())
        self.assertTrue(osp.exists(f'{self.tmp_dir.name}/test.bbox.json'))
