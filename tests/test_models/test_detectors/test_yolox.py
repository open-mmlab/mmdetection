# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmdet.models import YOLOX
from mmdet.registry import MODELS
from .utils import demo_mm_inputs, get_detector_cfg


class TestYOLOX(TestCase):

    def test_preprocess_data(self):
        model = get_detector_cfg('yolox/yolox_tiny_8x8_300e_coco.py')
        model.random_size_interval = 1
        model.random_size_range = (10, 10)
        model.input_size = (128, 128)
        model = MODELS.build(model)
        model.train()
        self.assertTrue(isinstance(model, YOLOX))

        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 128, 128]])
        batch_inputs, batch_data_samples = model.preprocess_data(packed_inputs)
        self.assertEqual(batch_inputs.shape, (2, 3, 128, 128))

        # resize after one iter
        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 128, 128]])
        batch_inputs, batch_data_samples = model.preprocess_data(packed_inputs)
        self.assertEqual(batch_inputs.shape, (2, 3, 320, 320))

        model.eval()
        packed_inputs = demo_mm_inputs(2, [[3, 128, 128], [3, 128, 128]])
        batch_inputs, batch_data_samples = model.preprocess_data(packed_inputs)
        self.assertEqual(batch_inputs.shape, (2, 3, 128, 128))
