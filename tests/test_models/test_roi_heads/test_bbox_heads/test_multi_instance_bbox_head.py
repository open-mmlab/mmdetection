# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.roi_heads.bbox_heads import MultiInstanceBBoxHead


class TestMultiInstanceBBoxHead(TestCase):

    def test_init(self):
        bbox_head = MultiInstanceBBoxHead(
            num_instance=2,
            with_refine=True,
            num_shared_fcs=2,
            in_channels=1,
            fc_out_channels=1,
            num_classes=4)
        self.assertTrue(bbox_head.shared_fcs_ref)
        self.assertTrue(bbox_head.fc_reg)
        self.assertTrue(bbox_head.fc_cls)
        self.assertEqual(len(bbox_head.shared_fcs), 2)
        self.assertEqual(len(bbox_head.fc_reg), 2)
        self.assertEqual(len(bbox_head.fc_cls), 2)

    def test_bbox_head_get_results(self):
        num_classes = 1
        num_instance = 2
        bbox_head = MultiInstanceBBoxHead(
            num_instance=num_instance,
            num_shared_fcs=2,
            reg_class_agnostic=True,
            num_classes=num_classes)
        s = 128
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]

        num_samples = 2
        rois = [torch.rand((num_samples, 5))]
        cls_scores = []
        bbox_preds = []
        for k in range(num_instance):
            cls_scores.append(torch.rand((num_samples, num_classes + 1)))
            bbox_preds.append(torch.rand((num_samples, 4)))
        cls_scores = [torch.cat(cls_scores, dim=1)]
        bbox_preds = [torch.cat(bbox_preds, dim=1)]

        # with nms
        rcnn_test_cfg = ConfigDict(
            nms=dict(type='nms', iou_threshold=0.5),
            score_thr=0.01,
            max_per_img=500)
        result_list = bbox_head.predict_by_feat(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas,
            rcnn_test_cfg=rcnn_test_cfg)

        self.assertLessEqual(
            len(result_list[0]), num_samples * num_instance * num_classes)
        self.assertIsInstance(result_list[0], InstanceData)
        self.assertEqual(result_list[0].bboxes.shape[1], 4)
        self.assertEqual(len(result_list[0].scores.shape), 1)
        self.assertEqual(len(result_list[0].labels.shape), 1)

        # without nms
        result_list = bbox_head.predict_by_feat(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas)

        self.assertIsInstance(result_list[0], InstanceData)
        self.assertEqual(len(result_list[0]), num_samples * num_instance)
        self.assertIsNone(result_list[0].get('label', None))

        # num_samples is 0
        num_samples = 0
        rois = [torch.rand((num_samples, 5))]
        cls_scores = []
        bbox_preds = []
        for k in range(num_instance):
            cls_scores.append(torch.rand((num_samples, num_classes + 1)))
            bbox_preds.append(torch.rand((num_samples, 4)))
        cls_scores = [torch.cat(cls_scores, dim=1)]
        bbox_preds = [torch.cat(bbox_preds, dim=1)]

        # with nms
        rcnn_test_cfg = ConfigDict(
            score_thr=0.,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        result_list = bbox_head.predict_by_feat(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas,
            rcnn_test_cfg=rcnn_test_cfg)

        self.assertIsInstance(result_list[0], InstanceData)
        self.assertEqual(len(result_list[0]), 0)
        self.assertEqual(result_list[0].bboxes.shape[1], 4)

        # without nms
        result_list = bbox_head.predict_by_feat(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas)

        self.assertIsInstance(result_list[0], InstanceData)
        self.assertEqual(len(result_list[0]), 0 * num_instance)
        self.assertIsNone(result_list[0].get('label', None))
