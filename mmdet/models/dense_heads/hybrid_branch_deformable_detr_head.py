# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import HEADS
from .detr_head import DETRHead


@HEADS.register_module()
class HybridBranchDeformableDETRHead(DETRHead):
    """Head of DeformDETR: Deformable DETR: Deformable Transformers for End-to-
    End Object Detection.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
        self,
        *args,
        num_query_one2one=300,
        k_one2many=5,
        with_box_refine=False,
        as_two_stage=False,
        mixed_selection=False,
        transformer=None,
        **kwargs,
    ):
        self.num_query_one2one = num_query_one2one
        self.k_one2many = k_one2many
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.mixed_selection = mixed_selection
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage

        super(HybridBranchDeformableDETRHead, self).__init__(
            *args, transformer=transformer, **kwargs
        )

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:

            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        elif self.mixed_selection:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)  #mixed query selection, MQS

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            one2one_cls_scores (Tensor): One-to-one branch outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            one2one_bbox_preds (Tensor): One-to-one branch sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            one2many_cls_scores (Tensor): One-to-many branch outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            one2many_bbox_preds (Tensor): One-to-many branch sigmoid outputs from the regression \
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
        batch_size = mlvl_feats[0].size(0) #2
        input_img_h, input_img_w = img_metas[0]["batch_input_shape"]  #704,939
        img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]  # 576,862
            img_masks[img_id, :img_h, :img_w] = 0  # 将mask中原始图像的部分设置为0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:])
                .to(torch.bool)
                .squeeze(0)
            )  # 将mask插值为不同scale的feature大小
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        
        # attn mask
        self_attn_mask = (
            torch.zeros([self.num_query, self.num_query,]).bool().to(feat.device)
        )  # num_query=1800
        self_attn_mask[self.num_query_one2one :, 0 : self.num_query_one2one] = True  # num_query_one-one=300
        self_attn_mask[0 : self.num_query_one2one, self.num_query_one2one :] = True

        if not self.as_two_stage or self.mixed_selection:
            query_embeds = self.query_embedding.weight  # 将query_embedding可学习权重赋值给query_embeds
        (   hs,
            init_reference,# 2,1800,4
            inter_references,# 6,2,1800,4
            enc_outputs_class,# 2,13805,80
            enc_outputs_coord,#2,13805,4
        ) = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches
            if self.with_box_refine
            else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
            decoder_self_attn_mask=[self_attn_mask, None],
        )
        hs = hs.permute(0, 2, 1, 3)  # 6,2,1800,256
        outputs_classes = []
        outputs_coords = []
        
        # import pdb; pdb.set_trace()
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)  # 2,1800,4
            outputs_class = self.cls_branches[lvl](hs[lvl])  # len(list)=2 (1800,80)
            tmp = self.reg_branches[lvl](hs[lvl])  # 2,1800,4
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # coordinate nm
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)  #6,2,1800,80
        outputs_coords = torch.stack(outputs_coords) #6,2,1800,4

        outputs_classes_one2one = outputs_classes[:, :, 0 : self.num_query_one2one, :]
        outputs_coords_one2one = outputs_coords[:, :, 0 : self.num_query_one2one, :]
        outputs_classes_one2many = outputs_classes[:, :, self.num_query_one2one :, :]
        outputs_coords_one2many = outputs_coords[:, :, self.num_query_one2one :, :]

        if self.as_two_stage:
            return (
                outputs_classes_one2one,
                outputs_coords_one2one,
                outputs_classes_one2many,
                outputs_coords_one2many,
                enc_outputs_class,
                enc_outputs_coord.sigmoid(),
            )
        else:
            return (
                outputs_classes_one2one,
                outputs_coords_one2one,
                outputs_classes_one2many,
                outputs_coords_one2many,
                None,
                None,
            )

    @force_fp32(
        apply_to=(
            "one2one_cls_scores_list",
            "one2one_bbox_preds_list",
            "one2many_cls_scores_list",
            "one2many_bbox_preds_list",
        )
    )
    def loss(
        self,
        one2one_cls_scores,
        one2one_bbox_preds,
        one2many_cls_scores,
        one2many_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """"Loss function.

        Args:
            one2one_cls_scores (Tensor): One-to-one branch classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            one2one_bbox_preds (Tensor): One-to-one branch sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            one2many_cls_scores (Tensor): One-to-many branch classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            one2many_bbox_preds (Tensor): One-to-many branch sigmoid regression
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
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports "
            f"for gt_bboxes_ignore setting to None."
        )
        import pdb; pdb.set_trace()
        num_dec_layers = len(one2one_cls_scores)  # 6
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        # for one2one branch
        one2one_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        one2one_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        one2one_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        # one2one losses
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single,
            one2one_cls_scores,
            one2one_bbox_preds,
            one2one_gt_bboxes_list,
            one2one_gt_labels_list,
            img_metas_list,
            one2one_gt_bboxes_ignore_list,
        )

        # for one2many branch
        one2many_gt_bboxes_list = []
        one2many_gt_labels_list = []
        for gt_bboxes in gt_bboxes_list:
            one2many_gt_bboxes_list.append(gt_bboxes.repeat(self.k_one2many, 1))

        for gt_labels in gt_labels_list:
            one2many_gt_labels_list.append(gt_labels.repeat(self.k_one2many))

        one2many_gt_bboxes_list = [
            one2many_gt_bboxes_list for _ in range(num_dec_layers)
        ]
        one2many_gt_labels_list = [
            one2many_gt_labels_list for _ in range(num_dec_layers)
        ]
        one2many_gt_bboxes_ignore_list = one2one_gt_bboxes_ignore_list

        # one2many losses
        losses_cls_one2many, losses_bbox_one2many, losses_iou_one2many = multi_apply(
            self.loss_single,
            one2many_cls_scores,
            one2many_bbox_preds,
            one2many_gt_bboxes_list,
            one2many_gt_labels_list,
            img_metas_list,
            one2many_gt_bboxes_ignore_list,
        )

        # sum up losses for two branches
        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i]) for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = self.loss_single(
                enc_cls_scores,
                enc_bbox_preds,
                gt_bboxes_list,
                binary_labels_list,
                img_metas,
                gt_bboxes_ignore,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_loss_iou"] = enc_losses_iou

        # loss from the last decoder layer, one2one+one2many
        loss_dict["loss_cls"] = losses_cls[-1] + losses_cls_one2many[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1] + losses_bbox_one2many[-1]
        loss_dict["loss_iou"] = losses_iou[-1] + losses_iou_one2many[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for (
            loss_cls_i,
            loss_bbox_i,
            loss_iou_i,
            loss_cls_i_one2many,
            loss_bbox_i_one2many,
            loss_iou_i_one2many,
        ) in zip(
            losses_cls[:-1],
            losses_bbox[:-1],
            losses_iou[:-1],
            losses_cls_one2many[:-1],
            losses_bbox_one2many[:-1],
            losses_iou_one2many[:-1],
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i + loss_cls_i_one2many
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = (
                loss_bbox_i + loss_bbox_i_one2many
            )
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i + loss_iou_i_one2many
            num_dec_layer += 1
        return loss_dict

    @force_fp32(
        apply_to=(
            "one2one_cls_scores_list",
            "one2one_bbox_preds_list",
            "one2many_cls_scores_list",
            "one2many_bbox_preds_list",
        )
    )
    def get_bboxes(
        self,
        one2one_cls_scores,
        one2one_bbox_preds,
        one2many_cls_scores,
        one2many_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        img_metas,
        rescale=False,
    ):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            one2one_cls_scores (Tensor): One2one branch classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            one2one_bbox_preds (Tensor): One2one branch sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            one2one_cls_scores (Tensor): One2many branch classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            one2one_bbox_preds (Tensor): One2many branch sigmoid regression
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
        cls_scores = one2one_cls_scores[-1]
        bbox_preds = one2one_bbox_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            proposals = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale
            )
            result_list.append(proposals)
        return result_list
