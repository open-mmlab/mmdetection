from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, ga_loc_target,
                        ga_shape_target, delta2bbox, multi_apply,
                        weighted_smoothl1, weighted_sigmoid_focal_loss,
                        weighted_cross_entropy, weighted_binary_cross_entropy,
                        bounded_iou_loss, multiclass_nms)
from mmdet.ops import DeformConv
from ..registry import HEADS
from ..utils import bias_init_with_prob


@HEADS.register_module
class GuidedAnchorHead(nn.Module):
    """Guided-Anchor-based head (GA-RPN, GA-RetinaNet, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        octave_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
    .cls_sigmoid_loss (bool): Whether to use sigmoid loss for classification.
            (softmax by default)
        cls_focal_loss (bool): Whether to use focal loss for classification.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 octave_base_scale=8,
                 scales_per_octave=3,
                 octave_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 anchoring_means=[.0, .0, .0, .0],
                 anchoring_stds=[1.0, 1.0, 1.0, 1.0],
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loc_focal_loss=True,
                 cls_sigmoid_loss=False,
                 cls_focal_loss=False):
        super(GuidedAnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.octave_scales = octave_base_scale * np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        self.approxs_per_octave = len(self.octave_scales) * len(octave_ratios)
        self.octave_ratios = octave_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.anchoring_means = anchoring_means
        self.anchoring_stds = anchoring_stds
        self.target_means = target_means
        self.target_stds = target_stds
        self.loc_focal_loss = loc_focal_loss
        assert self.loc_focal_loss, 'only focal loss is supported in loc'
        self.cls_sigmoid_loss = cls_sigmoid_loss
        self.cls_focal_loss = cls_focal_loss

        self.approx_generators = []
        self.base_approx_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.approx_generators.append(
                AnchorGenerator(anchor_base, self.octave_scales,
                                self.octave_ratios))
            self.base_approx_generators.append(
                AnchorGenerator(anchor_base, [self.octave_scales[0]], [1.0]))
        # one anchor per location
        self.num_anchors = 1
        if self.cls_sigmoid_loss:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes

        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                    1)
        deformable_groups = 4
        offset_channels = 3 * 3 * 2
        self.conv_offset = nn.Conv2d(
            self.num_anchors * 2,
            deformable_groups * offset_channels,
            1,
            bias=False)
        self.conv_adaption = DeformConv(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            padding=1,
            deformable_groups=deformable_groups)
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        # self.extra_rpn_conv = nn.Conv2d(
        #     self.feat_channels, self.feat_channels, 3, padding=1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)
        # normal_init(self.extra_rpn_conv, std=0.01)

        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)

    def forward_single(self, x):
        loc_pred = self.conv_loc(x)
        shape_pred = self.conv_shape(x)
        offset = self.conv_offset(shape_pred.detach())
        x = self.relu(self.conv_adaption(x, offset))
        # x = self.relu(self.extra_rpn_conv(x))
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, shape_preds, img_metas):
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
        multi_level_approxs = []
        multi_level_base_approxs = []
        for i in range(num_levels):
            approxs = self.approx_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            base_approxs = self.base_approx_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_approxs.append(approxs)
            multi_level_base_approxs.append(base_approxs)
        approxs_list = [multi_level_approxs for _ in range(num_imgs)]
        base_approxs_list = [multi_level_base_approxs for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        guided_anchors_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            multi_level_guided_anchors = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.approx_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                # calculate predicted anchors
                anchor_deltas = shape_preds[i][img_id].permute(
                    1, 2, 0).contiguous().view(-1, 2).detach()
                base_approxs = base_approxs_list[img_id][i]
                bbox_deltas = anchor_deltas.new_full(base_approxs.size(), 0)
                bbox_deltas[:, 2:] = anchor_deltas
                guided_anchors = delta2bbox(
                    base_approxs,
                    bbox_deltas,
                    self.anchoring_means,
                    self.anchoring_stds,
                    wh_ratio_clip=1e-6)
                multi_level_guided_anchors.append(guided_anchors)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
            guided_anchors_list.append(multi_level_guided_anchors)
        return (approxs_list, valid_flag_list, base_approxs_list,
                guided_anchors_list)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        if self.cls_sigmoid_loss:
            if self.cls_focal_loss:
                cls_criterion = weighted_sigmoid_focal_loss
            else:
                cls_criterion = weighted_binary_cross_entropy
        else:
            if self.cls_focal_loss:
                raise NotImplementedError
            else:
                cls_criterion = weighted_cross_entropy
        if self.cls_focal_loss:
            loss_cls = cls_criterion(
                cls_score,
                labels,
                label_weights,
                gamma=cfg.gamma,
                alpha=cfg.alpha,
                avg_factor=num_total_samples)
        else:
            loss_cls = cls_criterion(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls, loss_reg

    def loss_shape_single(self, shape_pred, bbox_anchors, bbox_gts,
                          anchor_weights, anchor_total_num):
        shape_pred = shape_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        bbox_anchors = bbox_anchors.contiguous().view(-1, 4)
        bbox_gts = bbox_gts.contiguous().view(-1, 4)
        anchor_weights = anchor_weights.contiguous().view(-1, 4)
        bbox_deltas = bbox_anchors.new_full(bbox_anchors.size(), 0)
        bbox_deltas[:, 2:] += shape_pred
        loss_shape = bounded_iou_loss(
            bbox_deltas,
            self.anchoring_means,
            self.anchoring_stds,
            bbox_anchors,
            bbox_gts,
            anchor_weights,
            beta=0.2,
            avg_factor=anchor_total_num)
        return loss_shape

    def loss_loc_single(self, loc_pred, loc_target, loc_weight, loc_avg_factor,
                        cfg):
        if self.loc_focal_loss:
            loss_loc = weighted_sigmoid_focal_loss(
                loc_pred.reshape(-1, 1),
                loc_target.reshape(-1, 1).long(),
                loc_weight.reshape(-1, 1),
                avg_factor=loc_avg_factor)
        else:
            loss_loc = weighted_binary_cross_entropy(
                loc_pred.reshape(-1, 1),
                loc_target.reshape(-1, 1).long(),
                loc_weight.reshape(-1, 1),
                avg_factor=loc_avg_factor)
        if hasattr(cfg, 'loc_weight'):
            loss_loc = loss_loc * cfg.loc_weight
        return loss_loc

    def loss(self,
             cls_scores,
             bbox_preds,
             shape_preds,
             loc_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.approx_generators)

        loc_targets, loc_weights, loc_avg_factor = ga_loc_target(
            gt_bboxes,
            featmap_sizes,
            self.octave_base_scale,
            self.anchor_strides,
            center_ratio=cfg.center_ratio,
            ignore_ratio=cfg.ignore_ratio)
        (approxs_list, valid_flag_list,
         base_approxs_list, guided_anchors_list) = self.get_anchors(
             featmap_sizes, shape_preds, img_metas)

        shape_targets = ga_shape_target(
            approxs_list, valid_flag_list, base_approxs_list, gt_bboxes,
            img_metas, self.approxs_per_octave, cfg)
        if shape_targets is None:
            return None
        (bbox_anchors_list, bbox_gts_list, anchor_weights_list,
         all_inside_flags, anchor_fg_num, anchor_ng_num) = shape_targets
        anchor_total_num = anchor_fg_num + anchor_ng_num

        sampling = False if self.cls_focal_loss else True
        label_channels = self.cls_out_channels if self.cls_sigmoid_loss else 1
        cls_reg_targets = anchor_target(
            guided_anchors_list,
            all_inside_flags,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos if self.cls_focal_loss else
                             num_total_pos + num_total_neg)
        losses_cls, losses_reg = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        losses_loc, = multi_apply(
            self.loss_loc_single,
            loc_preds,
            loc_targets,
            loc_weights,
            loc_avg_factor=loc_avg_factor,
            cfg=cfg)
        losses_shape, = multi_apply(
            self.loss_shape_single,
            shape_preds,
            bbox_anchors_list,
            bbox_gts_list,
            anchor_weights_list,
            anchor_total_num=anchor_total_num)
        return dict(
            loss_cls=losses_cls,
            loss_reg=losses_reg,
            loss_shape=losses_shape,
            loss_loc=losses_loc)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   shape_preds,
                   loc_preds,
                   img_metas,
                   cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(shape_preds) == len(
            loc_preds)
        num_levels = len(cls_scores)
        mlvl_anchors = [
            self.base_approx_generators[i].grid_anchors(
                cls_scores[i].size()[-2:], self.anchor_strides[i])
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
            shape_pred_list = [
                shape_preds[i][img_id].detach() for i in range(num_levels)
            ]
            loc_pred_list = [
                loc_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(
                cls_score_list, bbox_pred_list, shape_pred_list, loc_pred_list,
                mlvl_anchors, img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          shape_preds,
                          loc_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, shape_pred, loc_pred, anchors in zip(
                cls_scores, bbox_preds, shape_preds, loc_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size(
            )[-2:] == shape_pred.size()[-2:] == loc_pred.size()[-2:]
            loc_pred = loc_pred.sigmoid()
            loc_mask = loc_pred >= cfg.loc_filter_thr
            mask = loc_mask.permute(1, 2, 0).expand(
                loc_mask.size(0), loc_mask.size(1), self.num_anchors)
            mask = mask.contiguous().view(-1)
            mask_inds = mask.nonzero()
            if mask_inds.numel() == 0:
                continue
            else:
                mask_inds = mask_inds.squeeze()
            shape_pred = shape_pred.permute(1, 2, 0).reshape(-1, 2)
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.cls_sigmoid_loss:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = anchors[mask_inds, :]
            shape_pred = shape_pred[mask_inds, :]
            scores = scores[mask_inds, :]
            bbox_pred = bbox_pred[mask_inds, :]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.cls_sigmoid_loss:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                shape_pred = shape_pred[topk_inds, :]
            anchor_deltas = shape_pred.new_full((shape_pred.size(0), 4), 0)
            anchor_deltas[:, 2:] = shape_pred
            guided_anchors = delta2bbox(anchors, anchor_deltas,
                                        self.anchoring_means,
                                        self.anchoring_stds)
            bboxes = delta2bbox(guided_anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.cls_sigmoid_loss:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels
