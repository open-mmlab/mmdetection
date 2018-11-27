from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from mmdet.core import (AnchorGenerator, anchor_target, multi_apply,
                        delta2bbox, weighted_smoothl1,
                        weighted_sigmoid_focal_loss, multiclass_nms)
from ..utils import normal_init, bias_init_with_prob


class RetinaHead(nn.Module):
    """Head of RetinaNet.

            / cls_convs - retina_cls (3x3 conv)
    input -
            \ reg_convs - retina_reg (3x3 conv)

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Class number (including background).
        stacked_convs (int): Number of convolutional layers added for cls and
            reg branch.
        feat_channels (int): Number of channels for the RPN feature map.
        scales_per_octave (int): Number of anchor scales per octave.
        octave_base_scale (int): Base octave scale. Anchor scales are computed
            as `s*2^(i/n)`, for i in [0, n-1], where s is `octave_base_scale`
            and n is `scales_per_octave`.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 num_classes,
                 stacked_convs=4,
                 feat_channels=256,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(RetinaHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            anchor_scales = octave_scales * octave_base_scale
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))
        self.relu = nn.ReLU(inplace=True)
        self.num_anchors = int(
            len(self.anchor_ratios) * self.scales_per_octave)
        self.cls_out_channels = self.num_classes - 1
        self.bbox_pred_dim = 4

        self.stacked_convs = stacked_convs
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            self.cls_convs.append(
                nn.Conv2d(chn, feat_channels, 3, stride=1, padding=1))
            self.reg_convs.append(
                nn.Conv2d(chn, feat_channels, 3, stride=1, padding=1))
        self.retina_cls = nn.Conv2d(
            feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            stride=1,
            padding=1)
        self.retina_reg = nn.Conv2d(
            feat_channels,
            self.num_anchors * self.bbox_pred_dim,
            3,
            stride=1,
            padding=1)
        self.debug_imgs = None

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m, std=0.01)
        for m in self.reg_convs:
            normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = self.relu(cls_conv(cls_feat))
        for reg_conv in self.reg_convs:
            reg_feat = self.relu(reg_conv(reg_feat))
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_pos_samples, cfg):
        # classification loss
        labels = labels.contiguous().view(-1, self.cls_out_channels)
        label_weights = label_weights.contiguous().view(
            -1, self.cls_out_channels)
        cls_score = cls_score.permute(0, 2, 3, 1).contiguous().view(
            -1, self.cls_out_channels)
        loss_cls = weighted_sigmoid_focal_loss(
            cls_score,
            labels,
            label_weights,
            cfg.gamma,
            cfg.alpha,
            avg_factor=num_pos_samples)
        # regression loss
        bbox_targets = bbox_targets.contiguous().view(-1, 4)
        bbox_weights = bbox_weights.contiguous().view(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_pos_samples)
        return loss_cls, loss_reg

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
             cfg):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_labels_list=gt_labels,
            cls_out_channels=self.cls_out_channels,
            sampling=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        losses_cls, losses_reg = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_pos_samples=num_total_pos,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_reg=losses_reg)

    def get_det_bboxes(self,
                       cls_scores,
                       bbox_preds,
                       img_metas,
                       cfg,
                       rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            results = self._get_det_bboxes_single(
                cls_score_list, bbox_pred_list, mlvl_anchors, img_shape,
                scale_factor, cfg, rescale)
            result_list.append(results)
        return result_list

    def _get_det_bboxes_single(self,
                               cls_scores,
                               bbox_preds,
                               mlvl_anchors,
                               img_shape,
                               scale_factor,
                               cfg,
                               rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_proposals = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).contiguous().view(
                -1, self.cls_out_channels)
            scores = cls_score.sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).contiguous().view(-1, 4)
            proposals = delta2bbox(anchors, bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                maxscores, _ = scores.max(dim=1)
                _, topk_inds = maxscores.topk(cfg.nms_pre)
                proposals = proposals[topk_inds, :]
                scores = scores[topk_inds, :]
            mlvl_proposals.append(proposals)
            mlvl_scores.append(scores)
        mlvl_proposals = torch.cat(mlvl_proposals)
        if rescale:
            mlvl_proposals /= scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_proposals, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
