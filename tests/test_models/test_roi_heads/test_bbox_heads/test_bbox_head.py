# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.data import InstanceData

from mmdet.models.roi_heads.bbox_heads import (BBoxHead, Shared2FCBBoxHead,
                                               Shared4Conv1FCBBoxHead)


class TestBboxHead(TestCase):

    def test_init(self):
        # Shared2FCBBoxHead
        bbox_head = Shared2FCBBoxHead(
            in_channels=1, fc_out_channels=1, num_classes=4)
        assert bbox_head.fc_cls
        assert bbox_head.fc_reg
        assert len(bbox_head.shared_fcs) == 2

        # Shared4Conv1FCBBoxHead
        bbox_head = Shared4Conv1FCBBoxHead(
            in_channels=1, fc_out_channels=1, num_classes=4)
        assert bbox_head.fc_cls
        assert bbox_head.fc_reg
        assert len(bbox_head.shared_convs) == 4
        assert len(bbox_head.shared_fcs) == 1
        print()

    def test_bbox_head_get_results(self):
        num_classes = 6
        bbox_head = BBoxHead(reg_class_agnostic=True, num_classes=num_classes)
        s = 128
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]

        num_samples = 2
        rois = [torch.rand((num_samples, 5))]
        cls_scores = [torch.rand((num_samples, num_classes + 1))]
        bbox_preds = [torch.rand((num_samples, 4))]

        # with nms
        rcnn_test_cfg = ConfigDict(
            score_thr=0.,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        result_list = bbox_head.get_results(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas,
            rcnn_test_cfg=rcnn_test_cfg)

        assert len(result_list[0]) <= num_samples * num_classes
        assert isinstance(result_list[0], InstanceData)
        assert result_list[0].bboxes.shape[1] == 4
        assert len(result_list[0].scores.shape) == \
               len(result_list[0].labels.shape) == 1

        # without nms
        result_list = bbox_head.get_results(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas)

        assert isinstance(result_list[0], InstanceData)
        assert len(result_list[0]) == num_samples
        assert result_list[0].bboxes.shape == bbox_preds[0].shape
        assert result_list[0].scores.shape == cls_scores[0].shape
        assert result_list[0].get('label', None) is None

        # num_samples is 0
        num_samples = 0
        rois = [torch.rand((num_samples, 5))]
        cls_scores = [torch.rand((num_samples, num_classes + 1))]
        bbox_preds = [torch.rand((num_samples, 4))]

        # with nms
        rcnn_test_cfg = ConfigDict(
            score_thr=0.,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        result_list = bbox_head.get_results(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas,
            rcnn_test_cfg=rcnn_test_cfg)

        assert isinstance(result_list[0], InstanceData)
        assert len(result_list[0]) == 0
        assert result_list[0].bboxes.shape[1] == 4

        # without nms
        result_list = bbox_head.get_results(
            rois=tuple(rois),
            cls_scores=tuple(cls_scores),
            bbox_preds=tuple(bbox_preds),
            batch_img_metas=img_metas)

        assert isinstance(result_list[0], InstanceData)
        assert len(result_list[0]) == 0
        assert result_list[0].bboxes.shape == bbox_preds[0].shape
        assert result_list[0].scores.shape == cls_scores[0].shape
        assert result_list[0].get('label', None) is None
