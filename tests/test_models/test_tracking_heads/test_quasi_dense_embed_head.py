# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import mmengine
import torch
from mmengine.structures import InstanceData

from mmdet.models.tracking_heads import QuasiDenseEmbedHead
from mmdet.registry import TASK_UTILS


def _dummy_bbox_sampling(rpn_results_list, batch_gt_instances):
    """Create sample results that can be passed to Head.get_targets."""
    num_imgs = len(rpn_results_list)
    feat = torch.rand(1, 1, 3, 3)
    assign_config = dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        ignore_iof_thr=-1)
    sampler_config = dict(
        type='CombinedSampler',
        num=4,
        pos_fraction=0.5,
        neg_pos_ub=3,
        add_gt_as_proposals=True,
        pos_sampler=dict(type='InstanceBalancedPosSampler'),
        neg_sampler=dict(type='RandomSampler'))
    bbox_assigner = TASK_UTILS.build(assign_config)
    bbox_sampler = TASK_UTILS.build(sampler_config)

    sampling_results = []
    for i in range(num_imgs):
        assign_result = bbox_assigner.assign(rpn_results_list[i],
                                             batch_gt_instances[i])
        sampling_result = bbox_sampler.sample(
            assign_result,
            rpn_results_list[i],
            batch_gt_instances[i],
            feats=feat)
        sampling_results.append(sampling_result)

    return sampling_results


class TestQuasiDenseEmbedHead(TestCase):

    def test_quasi_dense_embed_head_loss(self):
        cfg = mmengine.Config(
            dict(
                num_convs=4,
                num_fcs=1,
                embed_channels=256,
                norm_cfg=dict(type='GN', num_groups=32),
                loss_track=dict(
                    type='MultiPosCrossEntropyLoss', loss_weight=0.25),
                loss_track_aux=dict(
                    type='MarginL2Loss',
                    neg_pos_ub=3,
                    pos_margin=0,
                    neg_margin=0.1,
                    hard_mining=True,
                    loss_weight=1.0)))

        embed_head = QuasiDenseEmbedHead(**cfg)

        key_feats = torch.rand(2, 256, 7, 7)
        ref_feats = key_feats
        rpn_results = InstanceData()
        rpn_results.labels = torch.LongTensor([1, 2])
        rpn_results.priors = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874],
             [23.6667, 23.8757, 238.6326, 151.8874]])
        rpn_results_list = [rpn_results]

        gt_instance = InstanceData()
        gt_instance.labels = torch.LongTensor([1, 2])
        gt_instance.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874],
             [23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instance.instances_id = torch.LongTensor([1, 2])
        batch_gt_instances = [gt_instance]

        sampling_results = _dummy_bbox_sampling(rpn_results_list,
                                                batch_gt_instances)
        gt_match_indices_list = [torch.Tensor([0, 1])]
        loss_track = embed_head.loss(key_feats, ref_feats, sampling_results,
                                     sampling_results, gt_match_indices_list)
        assert loss_track['loss_track'] >= 0, 'track loss should be zero'
        assert loss_track['loss_track_aux'] > 0, 'aux loss should be non-zero'

    def test_quasi_dense_embed_head_predict(self):
        cfg = mmengine.Config(
            dict(
                num_convs=4,
                num_fcs=1,
                embed_channels=256,
                norm_cfg=dict(type='GN', num_groups=32),
                loss_track=dict(
                    type='MultiPosCrossEntropyLoss', loss_weight=0.25),
                loss_track_aux=dict(
                    type='MarginL2Loss',
                    neg_pos_ub=3,
                    pos_margin=0,
                    neg_margin=0.1,
                    hard_mining=True,
                    loss_weight=1.0)))

        embed_head = QuasiDenseEmbedHead(**cfg)

        key_feats = torch.rand(2, 256, 7, 7)
        track_feats = embed_head.predict(key_feats)

        assert isinstance(track_feats, torch.Tensor)
        assert track_feats.size() == (2, 256)
