# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import mmengine
import torch
from mmengine.structures import InstanceData

from mmdet.models.tracking_heads import RoIEmbedHead
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
        type='RandomSampler',
        num=512,
        pos_fraction=0.25,
        neg_pos_ub=-1,
        add_gt_as_proposals=False)
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


class TestRoIEmbedHead(TestCase):

    def test_roi_embed_head_loss(self):
        """Test roi embed head loss when truth is non-empty."""
        cfg = mmengine.Config(
            dict(
                num_convs=2,
                num_fcs=2,
                roi_feat_size=7,
                in_channels=16,
                fc_out_channels=32))

        embed_head = RoIEmbedHead(**cfg)

        x = torch.rand(1, 16, 7, 7)
        ref_x = torch.rand(1, 16, 7, 7)
        num_x_per_img = [1]
        num_x_per_ref_img = [1]
        x_split, ref_x_split = embed_head.forward(x, ref_x, num_x_per_img,
                                                  num_x_per_ref_img)

        gt_instance_ids = [torch.LongTensor([2])]
        ref_gt_instance_ids = [torch.LongTensor([2])]

        rpn_results = InstanceData()
        rpn_results.labels = torch.LongTensor([2])
        rpn_results.priors = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        rpn_results_list = [rpn_results]

        gt_instance = InstanceData()
        gt_instance.labels = torch.LongTensor([2])
        gt_instance.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instance.instances_id = torch.LongTensor([2])
        batch_gt_instances = [gt_instance]

        sampling_results = _dummy_bbox_sampling(rpn_results_list,
                                                batch_gt_instances)

        gt_losses = embed_head.loss_by_feat(x_split, ref_x_split,
                                            sampling_results, gt_instance_ids,
                                            ref_gt_instance_ids)
        assert gt_losses['loss_match'] > 0, 'match loss should be non-zero'
        assert gt_losses[
            'match_accuracy'] >= 0, 'match accuracy should be non-zero or zero'

    def test_roi_embed_head_predict(self):
        cfg = mmengine.Config(
            dict(
                num_convs=2,
                num_fcs=2,
                roi_feat_size=7,
                in_channels=16,
                fc_out_channels=32))

        embed_head = RoIEmbedHead(**cfg)

        x = torch.rand(1, 16, 7, 7)
        ref_x = torch.rand(1, 16, 7, 7)
        similarity_logits = embed_head.predict(x, ref_x)

        assert isinstance(similarity_logits, list)
        assert len(similarity_logits) == 1
