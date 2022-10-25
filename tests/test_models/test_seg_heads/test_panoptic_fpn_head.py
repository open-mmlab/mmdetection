import unittest

import torch
from mmengine.structures import PixelData
from mmengine.testing import assert_allclose

from mmdet.models.seg_heads import PanopticFPNHead
from mmdet.structures import DetDataSample


class TestPanopticFPNHead(unittest.TestCase):

    def test_init_weights(self):
        head = PanopticFPNHead(
            num_things_classes=2,
            num_stuff_classes=2,
            in_channels=32,
            inner_channels=32)
        head.init_weights()
        assert_allclose(head.conv_logits.bias.data,
                        torch.zeros_like(head.conv_logits.bias.data))

    def test_loss(self):
        head = PanopticFPNHead(
            num_things_classes=2,
            num_stuff_classes=2,
            in_channels=32,
            inner_channels=32,
            start_level=0,
            end_level=1)
        x = [torch.rand((2, 32, 8, 8)), torch.rand((2, 32, 4, 4))]
        data_sample1 = DetDataSample()
        data_sample1.gt_sem_seg = PixelData(
            sem_seg=torch.randint(0, 4, (1, 7, 8)))
        data_sample2 = DetDataSample()
        data_sample2.gt_sem_seg = PixelData(
            sem_seg=torch.randint(0, 4, (1, 7, 8)))
        batch_data_samples = [data_sample1, data_sample2]
        results = head.loss(x, batch_data_samples)
        self.assertIsInstance(results, dict)

    def test_predict(self):
        head = PanopticFPNHead(
            num_things_classes=2,
            num_stuff_classes=2,
            in_channels=32,
            inner_channels=32,
            start_level=0,
            end_level=1)
        x = [torch.rand((2, 32, 8, 8)), torch.rand((2, 32, 4, 4))]
        img_meta1 = {
            'batch_input_shape': (16, 16),
            'img_shape': (14, 14),
            'ori_shape': (12, 12),
        }
        img_meta2 = {
            'batch_input_shape': (16, 16),
            'img_shape': (16, 16),
            'ori_shape': (16, 16),
        }
        batch_img_metas = [img_meta1, img_meta2]
        head.eval()
        with torch.no_grad():
            seg_preds = head.predict(x, batch_img_metas, rescale=False)
            self.assertTupleEqual(seg_preds[0].shape[-2:], (16, 16))
            self.assertTupleEqual(seg_preds[1].shape[-2:], (16, 16))

            seg_preds = head.predict(x, batch_img_metas, rescale=True)
            self.assertTupleEqual(seg_preds[0].shape[-2:], (12, 12))
            self.assertTupleEqual(seg_preds[1].shape[-2:], (16, 16))
