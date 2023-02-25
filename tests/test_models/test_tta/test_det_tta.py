# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import ConfigDict

from mmdet.models import DetTTAModel
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.testing import get_detector_cfg
from mmdet.utils import register_all_modules


class TestDetTTAModel(TestCase):

    def setUp(self):
        register_all_modules()

    def test_det_tta_model(self):

        detector_cfg = get_detector_cfg(
            'retinanet/retinanet_r18_fpn_1x_coco.py')
        cfg = ConfigDict(
            type='DetTTAModel',
            module=detector_cfg,
            tta_cfg=dict(
                nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))

        model: DetTTAModel = MODELS.build(cfg)

        imgs = []
        data_samples = []
        directions = ['horizontal', 'vertical']
        for i in range(12):
            flip_direction = directions[0] if i % 3 == 0 else directions[1]
            imgs.append(torch.randn(1, 3, 100 + 10 * i, 100 + 10 * i))
            data_samples.append([
                DetDataSample(
                    metainfo=dict(
                        ori_shape=(100, 100),
                        img_shape=(100 + 10 * i, 100 + 10 * i),
                        scale_factor=((100 + 10 * i) / 100,
                                      (100 + 10 * i) / 100),
                        flip=(i % 2 == 0),
                        flip_direction=flip_direction), )
            ])

        model.test_step(dict(inputs=imgs, data_samples=data_samples))
