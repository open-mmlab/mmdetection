import torch
import torch.nn as nn
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import FFN, MultiheadAttention, build_transformer
from .bbox_head import BBoxHead


@HEADS.register_module()
class DIIHead(BBoxHead):
    r"""Dynamic Instance Interactive Head for `Sparse R-CNN: End-to-End Object
    Detection with Learnable Proposals <https://arxiv.org/abs/2011.12450>`_"""

    def __init__(self,
                 num_classes=80,
                 num_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 hidden_channels=256,
                 dropout=0.0,
                 in_channels=256,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=7,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_iou=None,
                 **kwargs):  # TODO
        super(DIIHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            **kwargs)
        assert self.reg_class_agnostic is True
        self.loss_iou = build_loss(loss_iou)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.fp16_enabled = False
        self.self_attention = MultiheadAttention(hidden_channels, num_heads,
                                                 dropout)
        self.self_attention_norm = build_norm_layer(
            dict(type='LN'), hidden_channels)[1]

        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(
            dict(type='LN'), hidden_channels)[1]

        self.ffn = FFN(
            hidden_channels,
            feedforward_channels,
            num_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), hidden_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), hidden_channels)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(hidden_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(hidden_channels, self.num_classes + 1)

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), hidden_channels)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(hidden_channels, 4)

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        N, num_proposals = proposal_feat.shape[:2]
        # Self attention
        proposal_feat = proposal_feat.view(N, num_proposals,
                                           self.hidden_channels).permute(
                                               1, 0, 2)
        proposal_feat = self.self_attention_norm(
            self.self_attention(proposal_feat))

        # instance interactive
        proposal_feat = proposal_feat.view(
            num_proposals, N,
            self.hidden_channels).permute(1, 0,
                                          2).reshape(1, N * num_proposals,
                                                     self.hidden_channels)
        roi_feat = roi_feat.view(N * num_proposals, self.hidden_channels,
                                 -1).permute(2, 0, 1)
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        fc_feat = obj_feat.transpose(0, 1).reshape(N * num_proposals, -1)
        cls_feat = fc_feat.clone()
        reg_feat = fc_feat.clone()

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(N, num_proposals, -1)
        bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, -1)

        return cls_score, bbox_delta, obj_feat.view(N, num_proposals, -1)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                num_pos = pos_inds.sum()

                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=num_pos,
                )
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=num_pos,
                )
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            # TODO re_design _get_target_single
            #  and _get_targets in bbox_head
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        assert label_weights.sum() == 100
        return labels, label_weights, bbox_targets, bbox_weights

    # TODO: redisign bbox_head to support pos_inds and neg_inds
    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):

        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
