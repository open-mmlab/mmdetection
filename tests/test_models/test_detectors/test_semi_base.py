# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmengine.registry import MODELS
from parameterized import parameterized

from mmdet.testing import get_detector_cfg
from mmdet.utils import register_all_modules

register_all_modules()


class TestSemiBase(TestCase):

    @parameterized.expand([
        'soft_teacher/'
        'soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py',
    ])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        # backbone convert to ResNet18
        model.detector.backbone.depth = 18
        model.detector.neck.in_channels = [64, 128, 256, 512]
        model.detector.backbone.init_cfg = None

        model = MODELS.build(model)
        self.assertTrue(model.teacher.backbone)
        self.assertTrue(model.teacher.neck)
        self.assertTrue(model.teacher.rpn_head)
        self.assertTrue(model.teacher.roi_head)
        self.assertTrue(model.student.backbone)
        self.assertTrue(model.student.neck)
        self.assertTrue(model.student.rpn_head)
        self.assertTrue(model.student.roi_head)
