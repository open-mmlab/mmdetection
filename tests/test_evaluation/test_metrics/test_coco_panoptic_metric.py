import os
import os.path as osp
import tempfile
import unittest
from copy import deepcopy

import mmcv
import numpy as np
import torch
from mmengine.fileio import dump

from mmdet.evaluation import INSTANCE_OFFSET, CocoPanopticMetric

try:
    import panopticapi
except ImportError:
    panopticapi = None


class TestCocoPanopticMetric(unittest.TestCase):

    def _create_panoptic_gt_annotations(self, ann_file, seg_map_dir):
        categories = [{
            'id': 0,
            'name': 'person',
            'supercategory': 'person',
            'isthing': 1
        }, {
            'id': 1,
            'name': 'cat',
            'supercategory': 'cat',
            'isthing': 1
        }, {
            'id': 2,
            'name': 'dog',
            'supercategory': 'dog',
            'isthing': 1
        }, {
            'id': 3,
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
                'category_id': 2,
                'iscrowd': 0,
                'bbox': [50, 10, 10, 5],
                'area': 50
            }, {
                'id': 4,
                'category_id': 3,
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
        gt_bboxes = np.array(
            [[10, 10, 10, 40], [30, 10, 10, 40], [50, 10, 10, 5]],
            dtype=np.int64)
        for i in range(3):
            x, y, w, h = gt_bboxes[i]
            gt[y:y + h, x:x + w] = i + 1  # id starts from 1

        rgb_gt_seg_map = np.zeros(gt.shape + (3, ), dtype=np.uint8)
        rgb_gt_seg_map[:, :, 2] = gt // (256 * 256)
        rgb_gt_seg_map[:, :, 1] = gt % (256 * 256) // 256
        rgb_gt_seg_map[:, :, 0] = gt % 256

        img_path = osp.join(seg_map_dir, 'fake_name1.png')
        mmcv.imwrite(rgb_gt_seg_map[:, :, ::-1], img_path)
        dump(gt_json, ann_file)

        return gt_json

    def _create_panoptic_data_samples(self):
        # predictions
        # TP for background class, IoU=3576/4324=0.827
        # 2 the category id of the background class
        pred = np.zeros((60, 80), dtype=np.int64) + 2
        pred_bboxes = np.array(
            [
                [11, 11, 10, 40],  # TP IoU=351/449=0.78
                [38, 10, 10, 40],  # FP
                [51, 10, 10, 5]  # TP IoU=45/55=0.818
            ],
            dtype=np.int64)
        pred_labels = np.array([0, 0, 1], dtype=np.int64)
        for i in range(3):
            x, y, w, h = pred_bboxes[i]
            pred[y:y + h, x:x + w] = (i + 1) * INSTANCE_OFFSET + pred_labels[i]

        data_samples = [{
            'img_id':
            0,
            'ori_shape': (60, 80),
            'img_path':
            'xxx/fake_name1.jpg',
            'segments_info': [{
                'id': 1,
                'category': 0,
                'is_thing': 1
            }, {
                'id': 2,
                'category': 0,
                'is_thing': 1
            }, {
                'id': 3,
                'category': 1,
                'is_thing': 1
            }, {
                'id': 4,
                'category': 2,
                'is_thing': 0
            }],
            'seg_map_path':
            osp.join(self.gt_seg_dir, 'fake_name1.png'),
            'pred_panoptic_seg': {
                'sem_seg': torch.from_numpy(pred).unsqueeze(0)
            },
        }]

        return data_samples

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.gt_json_path = osp.join(self.tmp_dir.name, 'gt.json')
        self.gt_seg_dir = osp.join(self.tmp_dir.name, 'gt_seg')
        os.mkdir(self.gt_seg_dir)
        self._create_panoptic_gt_annotations(self.gt_json_path,
                                             self.gt_seg_dir)
        self.dataset_meta = {
            'classes': ('person', 'dog', 'wall'),
            'thing_classes': ('person', 'dog'),
            'stuff_classes': ('wall', )
        }
        self.target = {
            'coco_panoptic/PQ': 67.86874803219071,
            'coco_panoptic/SQ': 80.89770126158936,
            'coco_panoptic/RQ': 83.33333333333334,
            'coco_panoptic/PQ_th': 60.45252075318891,
            'coco_panoptic/SQ_th': 79.9959505972869,
            'coco_panoptic/RQ_th': 75.0,
            'coco_panoptic/PQ_st': 82.70120259019427,
            'coco_panoptic/SQ_st': 82.70120259019427,
            'coco_panoptic/RQ_st': 100.0
        }
        self.data_samples = self._create_panoptic_data_samples()

    def tearDown(self):
        self.tmp_dir.cleanup()

    @unittest.skipIf(panopticapi is not None, 'panopticapi is installed')
    def test_init(self):
        with self.assertRaises(RuntimeError):
            CocoPanopticMetric()

    @unittest.skipIf(panopticapi is None, 'panopticapi is not installed')
    def test_evaluate_without_json(self):
        # with tmpfile, without json
        metric = CocoPanopticMetric(
            ann_file=None,
            seg_prefix=self.gt_seg_dir,
            classwise=False,
            nproc=1,
            outfile_prefix=None)

        metric.dataset_meta = self.dataset_meta
        metric.process({}, deepcopy(self.data_samples))
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, self.target)

        # without tmpfile and json
        outfile_prefix = f'{self.tmp_dir.name}/test'
        metric = CocoPanopticMetric(
            ann_file=None,
            seg_prefix=self.gt_seg_dir,
            classwise=False,
            nproc=1,
            outfile_prefix=outfile_prefix)

        metric.dataset_meta = self.dataset_meta
        metric.process({}, deepcopy(self.data_samples))
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, self.target)

    @unittest.skipIf(panopticapi is None, 'panopticapi is not installed')
    def test_evaluate_with_json(self):
        # with tmpfile and json
        metric = CocoPanopticMetric(
            ann_file=self.gt_json_path,
            seg_prefix=self.gt_seg_dir,
            classwise=False,
            nproc=1,
            outfile_prefix=None)

        metric.dataset_meta = self.dataset_meta
        metric.process({}, deepcopy(self.data_samples))
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, self.target)

        # classwise
        metric = CocoPanopticMetric(
            ann_file=self.gt_json_path,
            seg_prefix=self.gt_seg_dir,
            classwise=True,
            nproc=1,
            outfile_prefix=None)
        metric.dataset_meta = self.dataset_meta
        metric.process({}, deepcopy(self.data_samples))
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, self.target)

        # without tmpfile, with json
        outfile_prefix = f'{self.tmp_dir.name}/test1'
        metric = CocoPanopticMetric(
            ann_file=self.gt_json_path,
            seg_prefix=self.gt_seg_dir,
            classwise=False,
            nproc=1,
            outfile_prefix=outfile_prefix)
        metric.dataset_meta = self.dataset_meta
        metric.process({}, deepcopy(self.data_samples))
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, self.target)

    @unittest.skipIf(panopticapi is None, 'panopticapi is not installed')
    def test_format_only(self):
        with self.assertRaises(AssertionError):
            metric = CocoPanopticMetric(
                ann_file=self.gt_json_path,
                seg_prefix=self.gt_seg_dir,
                classwise=False,
                nproc=1,
                format_only=True,
                outfile_prefix=None)

        outfile_prefix = f'{self.tmp_dir.name}/test'
        metric = CocoPanopticMetric(
            ann_file=self.gt_json_path,
            seg_prefix=self.gt_seg_dir,
            classwise=False,
            nproc=1,
            format_only=True,
            outfile_prefix=outfile_prefix)
        metric.dataset_meta = self.dataset_meta
        metric.process({}, deepcopy(self.data_samples))
        eval_results = metric.evaluate(size=1)
        self.assertDictEqual(eval_results, dict())
        self.assertTrue(osp.exists(f'{self.tmp_dir.name}/test.panoptic'))
        self.assertTrue(osp.exists(f'{self.tmp_dir.name}/test.panoptic.json'))
