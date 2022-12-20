# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, multi_apply)
from ..builder import HEADS
from ..losses import smooth_l1_loss
from .ascend_anchor_head import AscendAnchorHead
from .ssd_head import SSDHead
from ...utils import set_index


@HEADS.register_module()
class AscendSSDHead(SSDHead, AscendAnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Default: 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Dictionary to construct and config activation layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes=80,
                 in_channels=(512, 1024, 512, 256, 256, 256),
                 stacked_convs=0,
                 feat_channels=256,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 anchor_generator=dict(
                     type='SSDAnchorGenerator',
                     scale_major=False,
                     input_size=300,
                     strides=[8, 16, 32, 64, 100, 300],
                     ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
                     basesize_ratio_range=(0.1, 0.9)),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0],
                 ),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Xavier',
                     layer='Conv2d',
                     distribution='uniform',
                     bias=0)):
        super(AscendSSDHead, self).__init__(
            num_classes,
            in_channels,
            stacked_convs,
            feat_channels,
            use_depthwise,
            conv_cfg,
            norm_cfg,
            act_cfg,
            anchor_generator,
            bbox_coder,
            reg_decoded_bbox,
            train_cfg,
            test_cfg,
            init_cfg)

    def get_static_anchors(self, featmap_sizes, img_metas, device="cuda"):
        if not hasattr(self, 'static_anchors') or not hasattr(self, 'static_valid_flags'):
            static_anchors, static_valid_flags = self.get_anchors(featmap_sizes, img_metas, device)
            self.static_anchors = static_anchors
            self.static_valid_flags = static_valid_flags
        return self.static_anchors, self.static_valid_flags

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False,
                    return_level=True):
        return AscendAnchorHead.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            img_metas,
            gt_bboxes_ignore_list,
            gt_labels_list,
            label_channels,
            unmap_outputs,
            return_sampling_results,
            return_level,
        )

    def concat_loss(self,
                    concat_cls_score, concat_bbox_pred,
                    concat_anchor, concat_labels,
                    concat_label_weights,
                    concat_bbox_targets, concat_bbox_weights,
                    concat_pos_mask, concat_neg_mask,
                    num_total_samples):
        num_images, num_anchors, _ = concat_anchor.size()

        concat_loss_cls_all = F.cross_entropy(
            concat_cls_score.view((-1, self.cls_out_channels)), concat_labels.view(-1),
            reduction='none').view(concat_label_weights.size()) * concat_label_weights
        # # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        concat_num_pos_samples = torch.sum(concat_pos_mask, dim=1)
        concat_num_neg_samples = self.train_cfg.neg_pos_ratio * concat_num_pos_samples

        concat_num_neg_samples_max = torch.sum(concat_neg_mask, dim=1)
        concat_num_neg_samples = torch.min(concat_num_neg_samples, concat_num_neg_samples_max)

        concat_topk_loss_cls_neg, _ = torch.topk(concat_loss_cls_all * concat_neg_mask, k=num_anchors, dim=1)
        concat_loss_cls_pos = torch.sum(concat_loss_cls_all * concat_pos_mask, dim=1)

        anchor_index = torch.arange(end=num_anchors, dtype=torch.float, device=concat_anchor.device).view((1, -1))
        topk_loss_neg_mask = (anchor_index < concat_num_neg_samples.view(-1, 1)).float()

        concat_loss_cls_neg = torch.sum(concat_topk_loss_cls_neg * topk_loss_neg_mask, dim=1)
        loss_cls = (concat_loss_cls_pos + concat_loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            raise RuntimeError

        loss_bbox_all = smooth_l1_loss(
            concat_bbox_pred,
            concat_bbox_targets,
            concat_bbox_weights,
            reduction="none",
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        eps = torch.finfo(torch.float32).eps

        sum_dim = (i for i in range(1, len(loss_bbox_all.size())))
        loss_bbox = loss_bbox_all.sum(tuple(sum_dim)) / (num_total_samples + eps)
        return loss_cls[None], loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=True,
            return_level=False)
        if cls_reg_targets is None:
            return None

        (concat_labels, concat_label_weights, concat_bbox_targets, concat_bbox_weights, concat_pos_mask,
         concat_neg_mask, sampling_result, num_total_pos, num_total_neg, concat_anchors) = cls_reg_targets

        num_imgs = len(img_metas)
        concat_cls_score = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.cls_out_channels) for s in cls_scores
        ], 1)

        concat_bbox_pred = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for b in bbox_preds
        ], -2)

        concat_losses_cls, concat_losses_bbox = self.concat_loss(
            concat_cls_score, concat_bbox_pred,
            concat_anchors, concat_labels,
            concat_label_weights,
            concat_bbox_targets, concat_bbox_weights,
            concat_pos_mask, concat_neg_mask,
            num_total_pos)
        losses_cls = [concat_losses_cls[:, index_imgs] for index_imgs in range(num_imgs)]
        losses_bbox = [losses_bbox for losses_bbox in concat_losses_bbox]
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
