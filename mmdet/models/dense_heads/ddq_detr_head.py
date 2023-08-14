# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from ..layers import inverse_sigmoid
from ..losses import DDQAuxLoss
from ..utils import multi_apply
from .dino_head import DINOHead


@MODELS.register_module()
class DDQDETRHead(DINOHead):
    r"""Head of DDQDETR: Dense Distinct Query for
        End-to-End Object Detection.

    Code is modified from the `official github repo
        <https://github.com/jshilong/DDQ>`_.

    More details can be found in the `paper
        <https://arxiv.org/abs/2303.12776>`_ .

    Args:
        aux_num_pos (int): Number of positive targets assigned to a
            perdicted object. Defaults to 4.
    """

    def __init__(self, *args, aux_num_pos=4, **kwargs):
        super(DDQDETRHead, self).__init__(*args, **kwargs)
        self.aux_loss_for_dense = DDQAuxLoss(
            train_cfg=dict(
                assigner=dict(type='TopkHungarianAssigner', topk=aux_num_pos),
                alpha=1,
                beta=6))

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of aux head
        for dense queries."""
        super(DDQDETRHead, self)._init_layers()
        # If decoder `num_layers` = 6 and `as_two_stage` = True, then:
        #   1) 6 main heads are required for
        #       each decoder output of distinct queries.
        #   2) 1 main head is required for `output_memory` of distinct queries.
        #   3) 1 aux head is required for `output_memory` of dense queries,
        #       which is done by code below this comment.
        # So 8 heads are required in sum.
        # aux head for dense queries on encoder feature map
        self.cls_branches.append(copy.deepcopy(self.cls_branches[-1]))
        self.reg_branches.append(copy.deepcopy(self.reg_branches[-1]))

        # If decoder `num_layers` = 6 and `as_two_stage` = True, then:
        #   6 aux heads are required for each decoder output of dense queries.
        # So 8 + 6 = 14 heads and heads are requires in sum.
        # self.num_pred_layer is 7
        # aux head for dense queries in decoder
        self.aux_cls_branches = nn.ModuleList([
            copy.deepcopy(self.cls_branches[-1])
            for _ in range(self.num_pred_layer - 1)
        ])
        self.aux_reg_branches = nn.ModuleList([
            copy.deepcopy(self.reg_branches[-1])
            for _ in range(self.num_pred_layer - 1)
        ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m.bias, bias_init)
        for m in self.aux_cls_branches:
            nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.reg_branches:
            nn.init.constant_(m[-1].bias.data[2:], 0.0)

        for m in self.aux_reg_branches:
            constant_init(m[-1], 0, bias=0)

        for m in self.aux_reg_branches:
            nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries`, `num_queries` and `num_dense_queries`
                when `self.training` is `True`, else `num_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). Each reference has shape (bs,
                num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).

        Returns:
            tuple[Tensor]: results of head containing the following tensors.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries_total, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries_total, 4)
              with the last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        if self.training:
            num_dense = self.cache_dict['num_dense_queries']
        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            hidden_state = hidden_states[layer_id]
            if self.training:
                dense_hidden_state = hidden_state[:, -num_dense:]
                hidden_state = hidden_state[:, :-num_dense]

            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if self.training:
                dense_outputs_class = self.aux_cls_branches[layer_id](
                    dense_hidden_state)
                dense_tmp_reg_preds = self.aux_reg_branches[layer_id](
                    dense_hidden_state)
                outputs_class = torch.cat([outputs_class, dense_outputs_class],
                                          dim=1)
                tmp_reg_preds = torch.cat([tmp_reg_preds, dense_tmp_reg_preds],
                                          dim=1)

            if reference.shape[-1] == 4:
                tmp_reg_preds += reference
            else:
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def loss(self,
             hidden_states: Tensor,
             references: List[Tensor],
             enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Dict[str, int],
             aux_enc_outputs_class=None,
             aux_enc_outputs_coord=None) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries`, `num_queries` and `num_dense_queries`
                when `self.training` is `True`, else `num_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). Each reference has shape (bs,
                num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_outputs_class (Tensor): The top k classification score of
                each point on encoder feature map, has shape (bs, num_queries,
                cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generated from points
                with top k score, has shape (bs, num_queries, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.
            aux_enc_outputs_class (Tensor): The `dense_topk` classification
                score of each point on encoder feature map, has shape (bs,
                num_dense_queries, cls_out_channels).
                It is `None` when `self.training` is `False`.
            aux_enc_outputs_coord (Tensor): The proposal generated from points
                with `dense_topk` score, has shape (bs, num_dense_queries, 4)
                with the last dimension arranged as (cx, cy, w, h).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)

        aux_enc_outputs_coord = bbox_cxcywh_to_xyxy(aux_enc_outputs_coord)
        aux_enc_outputs_coord_list = []
        for img_id in range(len(aux_enc_outputs_coord)):
            det_bboxes = aux_enc_outputs_coord[img_id]
            img_shape = batch_img_metas[img_id]['img_shape']
            det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
            det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
            aux_enc_outputs_coord_list.append(det_bboxes)
        aux_enc_outputs_coord = torch.stack(aux_enc_outputs_coord_list)
        aux_loss = self.aux_loss_for_dense.loss(
            aux_enc_outputs_class.sigmoid(), aux_enc_outputs_coord,
            [item.bboxes for item in batch_gt_instances],
            [item.labels for item in batch_gt_instances], batch_img_metas)
        for k, v in aux_loss.items():
            losses[f'aux_enc_{k}'] = v

        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Bbox coordinates of all decoder
                layers. Each has shape (num_decoder_layers, bs,
                num_queries_total, 4) with normalized coordinate format
                (cx, cy, w, h).
            enc_cls_scores (Tensor): The top k score of each point on
                encoder feature map, has shape (bs, num_queries,
                cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generated from points
                with top k score, has shape (bs, num_queries, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        num_dense_queries = dn_meta['num_dense_queries']
        num_layer = all_layers_matching_bbox_preds.size(0)
        dense_all_layers_matching_cls_scores = all_layers_matching_cls_scores[:, :,  # noqa: E501
                                                                              -num_dense_queries:]  # noqa: E501
        dense_all_layers_matching_bbox_preds = all_layers_matching_bbox_preds[:, :,  # noqa: E501
                                                                              -num_dense_queries:]  # noqa: E501

        all_layers_matching_cls_scores = all_layers_matching_cls_scores[:, :, :  # noqa: E501
                                                                        -num_dense_queries]  # noqa: E501
        all_layers_matching_bbox_preds = all_layers_matching_bbox_preds[:, :, :  # noqa: E501
                                                                        -num_dense_queries]  # noqa: E501

        loss_dict = self.loss_for_distinct_queries(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)

        if enc_cls_scores is not None:

            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i

        for l_id in range(num_layer):
            cls_scores = dense_all_layers_matching_cls_scores[l_id].sigmoid()
            bbox_preds = dense_all_layers_matching_bbox_preds[l_id]

            bbox_preds = bbox_cxcywh_to_xyxy(bbox_preds)
            bbox_preds_list = []
            for img_id in range(len(bbox_preds)):
                det_bboxes = bbox_preds[img_id]
                img_shape = batch_img_metas[img_id]['img_shape']
                det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
                det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
                bbox_preds_list.append(det_bboxes)
            bbox_preds = torch.stack(bbox_preds_list)
            aux_loss = self.aux_loss_for_dense.loss(
                cls_scores, bbox_preds,
                [item.bboxes for item in batch_gt_instances],
                [item.labels for item in batch_gt_instances], batch_img_metas)
            for k, v in aux_loss.items():
                loss_dict[f'{l_id}_aux_{k}'] = v

        return loss_dict

    def loss_for_distinct_queries(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss of distinct queries, that is, excluding denoising
        and dense queries. Only select the distinct queries in decoder for
        loss.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries, cls_out_channels).
            all_layers_bbox_preds (Tensor): Bbox coordinates of all decoder
                layers. It has shape (num_decoder_layers, bs,
                num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image,
            e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self._loss_for_distinct_queries_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            [i for i in range(len(all_layers_bbox_preds))],
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def _loss_for_distinct_queries_single(self, cls_scores, bbox_preds, l_id,
                                          batch_gt_instances, batch_img_metas):
        """Calculate the loss for outputs from a single decoder layer of
        distinct queries, that is, excluding denoising and dense queries. Only
        select the distinct queries in decoder for loss.

        Args:
            cls_scores (Tensor): Classification scores of a single
                decoder layer, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Bbox coordinates of a single decoder
                layer. It has shape (bs, num_queries, 4) with the last
                dimension arranged as (cx, cy, w, h).
            l_id (int): Decoder layer index for these outputs.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image,
            e.g., image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        if 0 < l_id:
            batch_mask = [
                self.cache_dict['distinct_query_mask'][l_id - 1][
                    img_id * self.cache_dict['num_heads']][0]
                for img_id in range(num_imgs)
            ]
        else:
            batch_mask = [
                torch.ones(len(cls_scores[i]),
                           device=cls_scores.device).bool()
                for i in range(num_imgs)
            ]
        # only select the distinct queries in decoder for loss
        cls_scores_list = [
            cls_scores[i][batch_mask[i]] for i in range(num_imgs)
        ]
        bbox_preds_list = [
            bbox_preds[i][batch_mask[i]] for i in range(num_imgs)
        ]
        cls_scores = torch.cat(cls_scores_list)

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds_list):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = torch.cat(bbox_preds_list)
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def predict_by_feat(self,
                        layer_cls_scores: Tensor,
                        layer_bbox_preds: Tensor,
                        batch_img_metas: List[dict],
                        rescale: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            layer_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries, cls_out_channels).
            layer_bbox_preds (Tensor): Bbox coordinates of all decoder layers.
                Each has shape (num_decoder_layers, bs, num_queries, 4)
                with normalized coordinate format (cx, cy, w, h).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = layer_cls_scores[-1]
        bbox_preds = layer_bbox_preds[-1]

        num_imgs = cls_scores.size(0)
        # -1 is last layer input query mask

        batch_mask = [
            self.cache_dict['distinct_query_mask'][-1][
                img_id * self.cache_dict['num_heads']][0]
            for img_id in range(num_imgs)
        ]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id][batch_mask[img_id]]
            bbox_pred = bbox_preds[img_id][batch_mask[img_id]]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list
