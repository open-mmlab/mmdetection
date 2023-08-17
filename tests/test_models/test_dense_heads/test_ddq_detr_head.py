# Copyright (c) OpenMMLab. All rights reserved.
import json
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import DDQDETRHead
from mmdet.structures import DetDataSample


class TestDDQDETRHead(TestCase):

    def test_ddq_detr_head_loss(self):
        """Tests DDQDETRHead loss when truth is empty and non-empty."""
        num_classes = 2
        num_dn_queries = 10
        num_distinct_queries = 10
        dense_topk_ratio = 1.0
        num_decoder_layers = 2
        embed_dims = 256
        num_attention_heads = 8

        batch_data_samples = self.get_batch_data_samples()
        batch_size = len(batch_data_samples)

        batch_gt_instances = []
        batch_img_metas = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        max_num_target = max(
            len(data_sample.gt_instances)
            for data_sample in batch_data_samples)

        num_denoising_groups = max(num_dn_queries // max_num_target, 1)
        num_denoising_queries = num_denoising_groups * 2 * max_num_target

        num_dense_queries = int(num_distinct_queries * dense_topk_ratio)

        num_total_queries = (
            num_denoising_queries + num_distinct_queries + num_dense_queries)

        dn_meta = {
            'num_denoising_queries': num_denoising_queries,
            'num_denoising_groups': num_denoising_groups,
            'num_dense_queries': num_dense_queries
        }

        train_cfg = Config(
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ])))

        bbox_head = DDQDETRHead(
            num_classes=num_classes,
            sync_cls_avg_factor=True,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            share_pred_layer=False,
            num_pred_layer=1 + num_decoder_layers,
            as_two_stage=True,
            train_cfg=train_cfg,
            test_cfg=dict(max_per_img=300))

        bbox_head.cache_dict = dict(
            distinct_query_mask=list(
                torch.rand([
                    num_decoder_layers - 1, batch_size * num_attention_heads,
                    num_distinct_queries, num_distinct_queries
                ]) < 0.5),
            num_heads=num_attention_heads,
            num_dense_queries=num_dense_queries)

        # query
        hidden_states = torch.randn(
            [num_decoder_layers, batch_size, num_total_queries, embed_dims])
        # normalized cx, cy, w, h
        references = list(
            torch.rand(
                [1 + num_decoder_layers, batch_size, num_total_queries, 4]))
        all_layers_outputs_classes, all_layers_outputs_coords = \
            bbox_head.forward(hidden_states, references)

        # logits
        enc_outputs_class = torch.randn(
            [batch_size, num_distinct_queries, num_classes])

        # normalized cx, cy, w, h
        enc_outputs_coord = torch.rand([batch_size, num_distinct_queries, 4])

        # Test that empty ground truth encourages the network to predict
        # background
        empty_batch_gt_instances = []

        for _ in range(batch_size):
            gt_instances = InstanceData()
            gt_instances.labels = torch.LongTensor([])
            gt_instances.bboxes = torch.empty((0, 4))
            empty_batch_gt_instances.append(gt_instances)

        empty_gt_losses = bbox_head.loss_by_feat(all_layers_outputs_classes,
                                                 all_layers_outputs_coords,
                                                 enc_outputs_class,
                                                 enc_outputs_coord,
                                                 empty_batch_gt_instances,
                                                 batch_img_metas, dn_meta)

        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        losses = bbox_head.loss_by_feat(all_layers_outputs_classes,
                                        all_layers_outputs_coords,
                                        enc_outputs_class, enc_outputs_coord,
                                        batch_gt_instances, batch_img_metas,
                                        dn_meta)

        cls_loss = losses['loss_cls']
        box_loss = losses['loss_bbox']
        self.assertGreater(cls_loss.item(), 0, 'cls loss should be non-zero')
        self.assertGreater(box_loss.item(), 0, 'box loss should be non-zero')

    def get_batch_data_samples(self):
        """Generate batch data samples including model inputs and gt labels."""
        data_sample_file_path = 'tests/data/coco_batched_sample.json'

        with open(data_sample_file_path, 'r') as file_stream:
            data_sample_infos = json.load(file_stream)

        batch_data_samples = []

        for data_sample_info in data_sample_infos:
            data_sample = DetDataSample()

            metainfo = data_sample_info['metainfo']
            labels = data_sample_info['labels']
            bboxes = data_sample_info['bboxes']

            data_sample.set_metainfo(metainfo)

            gt_instances = InstanceData()
            gt_instances.labels = torch.LongTensor(labels)
            gt_instances.bboxes = torch.Tensor(bboxes)
            data_sample.gt_instances = gt_instances

            batch_data_samples.append(data_sample)

        return batch_data_samples
