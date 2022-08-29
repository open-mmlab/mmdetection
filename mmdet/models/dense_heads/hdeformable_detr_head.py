# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.deformable_detr_head import DeformableDETRHead
from mmdet.models.utils.transformer import inverse_sigmoid


@HEADS.register_module()
class HDeformableDETRHead(DeformableDETRHead):

    def __init__(self,
                 *args,
                 num_queries_one2one=300,
                 num_queries_one2many=0,
                 k_one2many=6,
                 lambda_one2many=1.0,
                 transformer=None,
                 **kwargs):
        self.num_queries_one2one = num_queries_one2one
        transformer['two_stage_num_proposals'] = (
            num_queries_one2one + num_queries_one2many)
        super().__init__(
            *args,
            num_query=num_queries_one2one + num_queries_one2many,
            transformer=transformer,
            **kwargs)
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage or self.mixed_selection:
            query_embeds = self.query_embedding.weight

        self_attn_mask = (
            torch.zeros([
                self.num_query,
                self.num_query,
            ]).bool().to(mlvl_feats[0].device))
        self_attn_mask[self.num_queries_one2one:,
                       0:self.num_queries_one2one, ] = True
        self_attn_mask[0:self.num_queries_one2one,
                       self.num_queries_one2one:, ] = True

        (hs, init_reference, inter_references, enc_outputs_class,
         enc_outputs_coord) = self.transformer(
             mlvl_feats,
             mlvl_masks,
             query_embeds,
             mlvl_positional_encodings,
             reg_branches=self.reg_branches if self.with_box_refine else None,
             # noqa:E501
             cls_branches=self.cls_branches if self.as_two_stage else None,
             # noqa:E501
             attn_masks=self_attn_mask)
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes_one2one.append(
                outputs_class[:, :self.num_queries_one2one])
            outputs_classes_one2many.append(
                outputs_class[:, self.num_queries_one2one:])
            outputs_coords_one2one.append(
                outputs_coord[:, :self.num_queries_one2one])
            outputs_coords_one2many.append(
                outputs_coord[:, self.num_queries_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        if self.as_two_stage:
            return outputs_classes_one2one, outputs_coords_one2one, \
                   outputs_classes_one2many, outputs_coords_one2many, \
                   enc_outputs_class, \
                   enc_outputs_coord.sigmoid()
        else:
            return outputs_classes_one2one, outputs_coords_one2one, \
                   outputs_classes_one2many, outputs_coords_one2many, \
                   None, None

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_one2one,
             all_bbox_preds_one2one,
             all_cls_scores_one2many,
             all_bbox_preds_one2many,
             enc_cls_scores,
             enc_bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores_one2one)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_bboxes_list_one2many = [[
            g.repeat(self.k_one2many, 1) for g in gt_bboxes_list
        ] for _ in range(num_dec_layers)]
        all_gt_labels_list_one2many = [[
            g.repeat(self.k_one2many, 1).view(-1) for g in gt_labels_list
        ] for _ in range(num_dec_layers)]
        if gt_bboxes_ignore is None:
            all_gt_bboxes_ignore_list_one2many = [
                gt_bboxes_ignore for _ in range(num_dec_layers)
            ]
        else:
            all_gt_bboxes_ignore_list_one2many = [[
                g.repeat(self.k_one2many, 1) if g is not None else None
                for g in gt_bboxes_ignore
            ] for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        (losses_cls_one2many, losses_bbox_one2many,
         losses_iou_one2many) = multi_apply(
             self.loss_single, all_cls_scores_one2many,
             all_bbox_preds_one2many, all_gt_bboxes_list_one2many,
             all_gt_labels_list_one2many, img_metas_list,
             all_gt_bboxes_ignore_list_one2many)
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores_one2one, all_bbox_preds_one2one,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_cls_one2many'] = (
            losses_cls_one2many[-1] * self.lambda_one2many)
        loss_dict['loss_bbox_one2many'] = (
            losses_bbox_one2many[-1] * self.lambda_one2many)
        loss_dict['loss_iou_one2many'] = (
            losses_iou_one2many[-1] * self.lambda_one2many)
        # loss from other decoder layers
        num_dec_layer = 0
        for (loss_cls_i, loss_bbox_i, loss_iou_i, loss_cls_i_one2many,
             loss_bbox_i_one2many, loss_iou_i_one2many) in zip(
                 losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1],
                 losses_cls_one2many[:-1], losses_bbox_one2many[:-1],
                 losses_iou_one2many[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[
                f'd{num_dec_layer}.loss_cls_one2many'] = loss_cls_i_one2many
            loss_dict[
                f'd{num_dec_layer}.loss_bbox_one2many'] = loss_bbox_i_one2many
            loss_dict[
                f'd{num_dec_layer}.loss_iou_one2many'] = loss_iou_i_one2many
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   all_cls_scores_one2many,
                   all_bbox_preds_one2many,
                   enc_cls_scores,
                   enc_bbox_preds,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.
        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)
        return result_list
