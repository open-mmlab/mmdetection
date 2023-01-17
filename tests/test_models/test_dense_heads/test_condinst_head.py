# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import CondInstBboxHead, CondInstMaskHead
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


class TestCondInstHead(TestCase):

    def test_condinst_bboxhead_loss(self):
        """Tests condinst bboxhead loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        condinst_bboxhead = CondInstBboxHead(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            norm_cfg=None)

        # Fcos head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in condinst_bboxhead.prior_generator.strides)
        cls_scores, bbox_preds, centernesses, param_preds =\
            condinst_bboxhead.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        gt_instances.masks = _rand_masks(0, gt_instances.bboxes.numpy(), s, s)

        empty_gt_losses = condinst_bboxhead.loss_by_feat(
            cls_scores, bbox_preds, centernesses, param_preds, [gt_instances],
            img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss and centerness loss should be zero
        empty_cls_loss = empty_gt_losses['loss_cls'].item()
        empty_box_loss = empty_gt_losses['loss_bbox'].item()
        empty_ctr_loss = empty_gt_losses['loss_centerness'].item()
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_ctr_loss, 0,
            'there should be no centerness loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        gt_instances.masks = _rand_masks(1, gt_instances.bboxes.numpy(), s, s)

        one_gt_losses = condinst_bboxhead.loss_by_feat(cls_scores, bbox_preds,
                                                       centernesses,
                                                       param_preds,
                                                       [gt_instances],
                                                       img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].item()
        onegt_box_loss = one_gt_losses['loss_bbox'].item()
        onegt_ctr_loss = one_gt_losses['loss_centerness'].item()
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(onegt_ctr_loss, 0,
                           'centerness loss should be non-zero')

        # Test the `center_sampling` works fine.
        condinst_bboxhead.center_sampling = True
        ctrsamp_losses = condinst_bboxhead.loss_by_feat(
            cls_scores, bbox_preds, centernesses, param_preds, [gt_instances],
            img_metas)
        ctrsamp_cls_loss = ctrsamp_losses['loss_cls'].item()
        ctrsamp_box_loss = ctrsamp_losses['loss_bbox'].item()
        ctrsamp_ctr_loss = ctrsamp_losses['loss_centerness'].item()
        self.assertGreater(ctrsamp_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(ctrsamp_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(ctrsamp_ctr_loss, 0,
                           'centerness loss should be non-zero')

        # Test the `norm_on_bbox` works fine.
        condinst_bboxhead.norm_on_bbox = True
        normbox_losses = condinst_bboxhead.loss_by_feat(
            cls_scores, bbox_preds, centernesses, param_preds, [gt_instances],
            img_metas)
        normbox_cls_loss = normbox_losses['loss_cls'].item()
        normbox_box_loss = normbox_losses['loss_bbox'].item()
        normbox_ctr_loss = normbox_losses['loss_centerness'].item()
        self.assertGreater(normbox_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(normbox_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(normbox_ctr_loss, 0,
                           'centerness loss should be non-zero')

    def test_condinst_maskhead_loss(self):
        """Tests condinst maskhead loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        condinst_bboxhead = CondInstBboxHead(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            norm_cfg=None)

        mask_feature_head = _fake_mask_feature_head()
        condinst_maskhead = CondInstMaskHead(
            mask_feature_head=mask_feature_head,
            loss_mask=dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                eps=5e-6,
                loss_weight=1.0))

        # Fcos head expects a multiple levels of features per image
        feats = []
        for i in range(len(condinst_bboxhead.strides)):
            feats.append(
                torch.rand(1, 1, s // (2**(i + 3)), s // (2**(i + 3))))
        feats = tuple(feats)
        cls_scores, bbox_preds, centernesses, param_preds =\
            condinst_bboxhead.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        gt_instances.masks = _rand_masks(0, gt_instances.bboxes.numpy(), s, s)

        _ = condinst_bboxhead.loss_by_feat(cls_scores, bbox_preds,
                                           centernesses, param_preds,
                                           [gt_instances], img_metas)
        # When truth is empty then all mask loss
        # should be zero for random inputs
        positive_infos = condinst_bboxhead.get_positive_infos()
        mask_outs = condinst_maskhead.forward(feats, positive_infos)
        empty_gt_mask_losses = condinst_maskhead.loss_by_feat(
            *mask_outs, [gt_instances], img_metas, positive_infos)
        loss_mask = empty_gt_mask_losses['loss_mask']
        self.assertEqual(loss_mask, 0, 'mask loss should be zero')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        gt_instances.masks = _rand_masks(1, gt_instances.bboxes.numpy(), s, s)

        _ = condinst_bboxhead.loss_by_feat(cls_scores, bbox_preds,
                                           centernesses, param_preds,
                                           [gt_instances], img_metas)
        positive_infos = condinst_bboxhead.get_positive_infos()
        mask_outs = condinst_maskhead.forward(feats, positive_infos)
        one_gt_mask_losses = condinst_maskhead.loss_by_feat(
            *mask_outs, [gt_instances], img_metas, positive_infos)
        loss_mask = one_gt_mask_losses['loss_mask']
        self.assertGreater(loss_mask, 0, 'mask loss should be nonzero')
