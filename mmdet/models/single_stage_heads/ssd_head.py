from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (AnchorGenerator, anchor_target, multi_apply,
                        delta2bbox, weighted_smoothl1, multiclass_nms)


class SSDHead(nn.Module):
    """Head of RetinaNet.

            / conf_layers - retina_cls (3x3 conv)
    input -
            \ loc_layers - retina_reg (3x3 conv)

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Class number (including background).
        stacked_convs (int): Number of convolutional layers added for cls and
            reg branch.
        feat_channels (int): Number of channels for the RPN feature map.
    """

    def __init__(self,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 num_classes=21,
                 anchor_strides=(8, 16, 32, 64, 100, 300),
                 min_sizes=(30, 60, 111, 162, 213, 264),
                 max_sizes=(60, 111, 162, 213, 264, 315),
                 aspect_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0)):
        super(SSDHead, self).__init__()
        # construct head
        num_anchors = [len(ratios) * 2 + 2 for ratios in aspect_ratios]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        loc_layers = []
        conf_layers = []
        for k, v in enumerate(in_channels):
            loc_layers += [nn.Conv2d(in_channels[k], num_anchors[k] * 4,
                                     kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(in_channels[k], num_anchors[k] *
                                      num_classes, kernel_size=3, padding=1)]
        self.loc_layers = nn.ModuleList(loc_layers)
        self.conf_layers = nn.ModuleList(conf_layers)

        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            s_k = base_size / 300
            s_k_prime = np.sqrt(s_k * (max_sizes[k] / 300))
            scales = [1., s_k_prime / s_k]  # based on s_k
            stride = anchor_strides[k]
            anchor_ratios = [1.]
            for r in aspect_ratios[k]:
                anchor_ratios += [1 / r, r]  # 4 or 6 ratio
            ctr = ((stride - 1) / 2, (stride - 1) / 2)
            anchor_generator = AnchorGenerator(base_size, scales,
                                               anchor_ratios,
                                               scale_major=False,
                                               ctr=ctr,
                                               clamp_size=300)
            indices = list(range(len(anchor_ratios)))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(
                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)

        self.target_means = target_means
        self.target_stds = target_stds

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, 'bias') is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(
                feats, self.loc_layers, self.conf_layers):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

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
        loss_cls_all = F.cross_entropy(cls_score, labels,
                                       reduction='none') * label_weights
        pos_label_inds = (labels > 0).nonzero().view(-1)
        neg_label_inds = (labels == 0).nonzero().view(-1)

        num_sample_pos = pos_label_inds.size(0)
        num_sample_neg = cfg.neg_pos_ratio * num_sample_pos
        if num_sample_neg > neg_label_inds.size(0):
            num_sample_neg = neg_label_inds.size(0)
        topk_loss_cls_neg, topk_loss_cls_neg_inds = \
            loss_cls_all[neg_label_inds].topk(num_sample_neg)
        loss_cls_pos = loss_cls_all[pos_label_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_pos_samples

        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_pos_samples)
        return loss_cls[None], loss_reg

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
            sampling=False,
            need_unmap=False)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([s.permute(0, 2, 3, 1).contiguous().view(
            num_images, -1, self.cls_out_channels) for s in cls_scores], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(
            label_weights_list, -1).view(num_images, -1)
        all_bbox_preds = torch.cat([b.permute(0, 2, 3, 1).contiguous().view(
            num_images, -1, 4) for b in bbox_preds], -2)
        all_bbox_targets = torch.cat(
            bbox_targets_list, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(
            bbox_weights_list, -2).view(num_images, -1, 4)

        losses_cls, losses_reg = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
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
            scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).contiguous().view(-1, 4)
            proposals = delta2bbox(anchors, bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            mlvl_proposals.append(proposals)
            mlvl_scores.append(scores)
        mlvl_proposals = torch.cat(mlvl_proposals)
        if rescale:
            mlvl_proposals /= mlvl_proposals.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        det_bboxes, det_labels = multiclass_nms(mlvl_proposals, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
