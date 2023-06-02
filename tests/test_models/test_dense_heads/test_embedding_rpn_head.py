# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import EmbeddingRPNHead
from mmdet.structures import DetDataSample


class TestEmbeddingRPNHead(TestCase):

    def test_init(self):
        """Test init rpn head."""
        rpn_head = EmbeddingRPNHead(
            num_proposals=100, proposal_feature_channel=256)
        rpn_head.init_weights()
        self.assertTrue(rpn_head.init_proposal_bboxes)
        self.assertTrue(rpn_head.init_proposal_features)

    def test_loss_and_predict(self):
        s = 256
        img_meta = {
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }
        rpn_head = EmbeddingRPNHead(
            num_proposals=100, proposal_feature_channel=256)

        feats = [
            torch.rand(2, 1, s // (2**(i + 2)), s // (2**(i + 2)))
            for i in range(5)
        ]

        data_sample = DetDataSample()
        data_sample.set_metainfo(img_meta)

        # test predict
        result_list = rpn_head.predict(feats, [data_sample])
        self.assertTrue(isinstance(result_list, list))
        self.assertTrue(isinstance(result_list[0], InstanceData))

        # test loss_and_predict
        result_list = rpn_head.loss_and_predict(feats, [data_sample])
        self.assertTrue(isinstance(result_list, tuple))
        self.assertTrue(isinstance(result_list[0], dict))
        self.assertEqual(len(result_list[0]), 0)
        self.assertTrue(isinstance(result_list[1], list))
        self.assertTrue(isinstance(result_list[1][0], InstanceData))

        # test loss
        with pytest.raises(NotImplementedError):
            rpn_head.loss(feats, [data_sample])
