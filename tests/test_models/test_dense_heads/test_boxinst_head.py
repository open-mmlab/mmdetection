# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine import MessageHub
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import BoxInstBboxHead, BoxInstMaskHead
from mmdet.structures.mask import BitmapMasks


def _rand_masks(num_items, bboxes, img_w, img_h):
    rng = np.random.RandomState(0)
    masks = np.zeros((num_items, img_h, img_w), dtype=np.float32)
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        mask = (rng.rand(1, bbox[3] - bbox[1], bbox[2] - bbox[0]) >
                0.3).astype(np.int64)
        masks[i:i + 1, bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask
    return BitmapMasks(masks, height=img_h, width=img_w)


def _fake_mask_feature_head():
    mask_feature_head = ConfigDict(
        in_channels=1,
        feat_channels=1,
        start_level=0,
        end_level=2,
        out_channels=8,
        mask_stride=8,
        num_stacked_convs=4,
        norm_cfg=dict(type='BN', requires_grad=True))
    return mask_feature_head


class TestBoxInstHead(TestCase):

    def test_boxinst_maskhead_loss(self):
        """Tests boxinst maskhead loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        boxinst_bboxhead = BoxInstBboxHead(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            norm_cfg=None)

        mask_feature_head = _fake_mask_feature_head()
        boxinst_maskhead = BoxInstMaskHead(
            mask_feature_head=mask_feature_head,
            loss_mask=dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                eps=5e-6,
                loss_weight=1.0))

        # Fcos head expects a multiple levels of features per image
        feats = []
        for i in range(len(boxinst_bboxhead.strides)):
            feats.append(
                torch.rand(1, 1, s // (2**(i + 3)), s // (2**(i + 3))))
        feats = tuple(feats)
        cls_scores, bbox_preds, centernesses, param_preds =\
            boxinst_bboxhead.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        gt_instances.masks = _rand_masks(0, gt_instances.bboxes.numpy(), s, s)
        gt_instances.pairwise_masks = _rand_masks(
            0, gt_instances.bboxes.numpy(), s // 4, s // 4).to_tensor(
                dtype=torch.float32,
                device='cpu').unsqueeze(1).repeat(1, 8, 1, 1)
        message_hub = MessageHub.get_instance('runtime_info')
        message_hub.update_info('iter', 1)
        _ = boxinst_bboxhead.loss_by_feat(cls_scores, bbox_preds, centernesses,
                                          param_preds, [gt_instances],
                                          img_metas)
        # When truth is empty then all mask loss
        # should be zero for random inputs
        positive_infos = boxinst_bboxhead.get_positive_infos()
        mask_outs = boxinst_maskhead.forward(feats, positive_infos)
        empty_gt_mask_losses = boxinst_maskhead.loss_by_feat(
            *mask_outs, [gt_instances], img_metas, positive_infos)
        loss_mask_project = empty_gt_mask_losses['loss_mask_project']
        loss_mask_pairwise = empty_gt_mask_losses['loss_mask_pairwise']
        self.assertEqual(loss_mask_project, 0,
                         'mask project loss should be zero')
        self.assertEqual(loss_mask_pairwise, 0,
                         'mask pairwise loss should be zero')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor([[0.111, 0.222, 25.6667, 29.8757]])
        gt_instances.labels = torch.LongTensor([2])
        gt_instances.masks = _rand_masks(1, gt_instances.bboxes.numpy(), s, s)
        gt_instances.pairwise_masks = _rand_masks(
            1, gt_instances.bboxes.numpy(), s // 4, s // 4).to_tensor(
                dtype=torch.float32,
                device='cpu').unsqueeze(1).repeat(1, 8, 1, 1)

        _ = boxinst_bboxhead.loss_by_feat(cls_scores, bbox_preds, centernesses,
                                          param_preds, [gt_instances],
                                          img_metas)
        positive_infos = boxinst_bboxhead.get_positive_infos()
        mask_outs = boxinst_maskhead.forward(feats, positive_infos)
        one_gt_mask_losses = boxinst_maskhead.loss_by_feat(
            *mask_outs, [gt_instances], img_metas, positive_infos)
        loss_mask_project = one_gt_mask_losses['loss_mask_project']
        loss_mask_pairwise = one_gt_mask_losses['loss_mask_pairwise']
        self.assertGreater(loss_mask_project, 0,
                           'mask project loss should be nonzero')
        self.assertGreater(loss_mask_pairwise, 0,
                           'mask pairwise loss should be nonzero')
