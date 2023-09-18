# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from torch import Tensor

from mmdet.models.dense_heads.dino_head import DINOHead
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList


@MODELS.register_module()
class HybridDINOHead(DINOHead):
    """Head of the Hybrid Matching."""

    def __init__(self,
                 *args,
                 num_query_one2one: int = 900,
                 k_one2many: int = 2,
                 **kwargs) -> None:
        self.num_query_one2one = num_query_one2one
        self.k_one2many = k_one2many
        super().__init__(*args, **kwargs)

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
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
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
        # train: num_denoising_queries + num_query_one2one
        # + num_query_one2many
        num_query_one2one = dn_meta[
            'num_denoising_queries'] + self.num_query_one2one
        outputs_classes_one2one = \
            all_layers_cls_scores[:, :, 0: num_query_one2one, :]
        outputs_coords_one2one = \
            all_layers_bbox_preds[:, :, 0: num_query_one2one, :]
        # hybrid-matching part
        outputs_classes_one2many = \
            all_layers_cls_scores[:, :, num_query_one2one:, :]
        outputs_coords_one2many = \
            all_layers_bbox_preds[:, :, num_query_one2one:, :]

        loss_dict = super(HybridDINOHead, self).loss_by_feat(
            outputs_classes_one2one, outputs_coords_one2one, enc_cls_scores,
            enc_bbox_preds, batch_gt_instances, batch_img_metas, dn_meta,
            batch_gt_instances_ignore)

        o2m_batch_gt_instances = []
        for gt_instance in batch_gt_instances:
            bboxes = gt_instance.bboxes.repeat(self.k_one2many, 1)
            labels = gt_instance.labels.repeat(self.k_one2many)
            new_gt_instance = gt_instance.new(bboxes=bboxes, labels=labels)
            o2m_batch_gt_instances.append(new_gt_instance)

        losses_cls_o2m, losses_bbox_o2m, losses_iou_o2m = multi_apply(
            self.loss_by_feat_single,
            outputs_classes_one2many,
            outputs_coords_one2many,
            batch_gt_instances=o2m_batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict['loss_cls_o2m'] = losses_cls_o2m[-1]
        loss_dict['loss_bbox_o2m'] = losses_bbox_o2m[-1]
        loss_dict['loss_iou_o2m'] = losses_iou_o2m[-1]
        for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                enumerate(zip(losses_cls_o2m[:-1], losses_bbox_o2m[:-1],
                              losses_iou_o2m[:-1])):
            loss_dict[f'd{num_dec_layer}.loss_cls_o2m'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_o2m'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou_o2m'] = loss_iou_i
        return loss_dict
