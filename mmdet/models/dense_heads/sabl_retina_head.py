# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from .guided_anchor_head import GuidedAnchorHead


@HEADS.register_module()
class SABLRetinaHead(BaseDenseHead, BBoxTestMixin):
    """Side-Aware Boundary Localization (SABL) for RetinaNet.

    The anchor generation, assigning and sampling in SABLRetinaHead
    are the same as GuidedAnchorHead for guided anchoring.

    Please refer to https://arxiv.org/abs/1912.04260 for more details.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of Convs for classification \
            and regression branches. Defaults to 4.
        feat_channels (int): Number of hidden channels. \
            Defaults to 256.
        approx_anchor_generator (dict): Config dict for approx generator.
        square_anchor_generator (dict): Config dict for square generator.
        conv_cfg (dict): Config dict for ConvModule. Defaults to None.
        norm_cfg (dict): Config dict for Norm Layer. Defaults to None.
        bbox_coder (dict): Config dict for bbox coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (dict): Training config of SABLRetinaHead.
        test_cfg (dict): Testing config of SABLRetinaHead.
        loss_cls (dict): Config of classification loss.
        loss_bbox_cls (dict): Config of classification loss for bbox branch.
        loss_bbox_reg (dict): Config of regression loss for bbox branch.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 feat_channels=256,
                 approx_anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 square_anchor_generator=dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     scales=[4],
                     strides=[8, 16, 32, 64, 128]),
                 conv_cfg=None,
                 norm_cfg=None,
                 bbox_coder=dict(
                     type='BucketingBBoxCoder',
                     num_buckets=14,
                     scale_factor=3.0),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.5),
                 loss_bbox_reg=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.5),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01))):
        super(SABLRetinaHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_buckets = bbox_coder['num_buckets']
        self.side_num = int(np.ceil(self.num_buckets / 2))

        assert (approx_anchor_generator['octave_base_scale'] ==
                square_anchor_generator['scales'][0])
        assert (approx_anchor_generator['strides'] ==
                square_anchor_generator['strides'])

        self.approx_anchor_generator = build_anchor_generator(
            approx_anchor_generator)
        self.square_anchor_generator = build_anchor_generator(
            square_anchor_generator)
        self.approxs_per_octave = (
            self.approx_anchor_generator.num_base_anchors[0])

        # one anchor per location
        self.num_anchors = 1
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.reg_decoded_bbox = reg_decoded_bbox

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_cls = build_loss(loss_bbox_cls)
        self.loss_bbox_reg = build_loss(loss_bbox_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.retina_bbox_reg = nn.Conv2d(
            self.feat_channels, self.side_num * 4, 3, padding=1)
        self.retina_bbox_cls = nn.Conv2d(
            self.feat_channels, self.side_num * 4, 3, padding=1)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_cls_pred = self.retina_bbox_cls(reg_feat)
        bbox_reg_pred = self.retina_bbox_reg(reg_feat)
        bbox_pred = (bbox_cls_pred, bbox_reg_pred)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get squares according to feature map sizes and guided anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: square approxs of each image
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # squares for one time
        multi_level_squares = self.square_anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        squares_list = [multi_level_squares for _ in range(num_imgs)]

        return squares_list

    def get_target(self,
                   approx_list,
                   inside_flag_list,
                   square_list,
                   gt_bboxes_list,
                   img_metas,
                   gt_bboxes_ignore_list=None,
                   gt_labels_list=None,
                   label_channels=None,
                   sampling=True,
                   unmap_outputs=True):
        """Compute bucketing targets.
        Args:
            approx_list (list[list]): Multi level approxs of each image.
            inside_flag_list (list[list]): Multi level inside flags of each
                image.
            square_list (list[list]): Multi level squares of each image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): ignore list of gt bboxes.
            gt_bboxes_list (list[Tensor]): Gt bboxes of each image.
            label_channels (int): Channel of label.
            sampling (bool): Sample Anchors or not.
            unmap_outputs (bool): unmap outputs or not.

        Returns:
            tuple: Returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_cls_targets_list (list[Tensor]): BBox cls targets of \
                    each level.
                - bbox_cls_weights_list (list[Tensor]): BBox cls weights of \
                    each level.
                - bbox_reg_targets_list (list[Tensor]): BBox reg targets of \
                    each level.
                - bbox_reg_weights_list (list[Tensor]): BBox reg weights of \
                    each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        num_imgs = len(img_metas)
        assert len(approx_list) == len(inside_flag_list) == len(
            square_list) == num_imgs
        # anchor number of multi levels
        num_level_squares = [squares.size(0) for squares in square_list[0]]
        # concat all level anchors and flags to a single tensor
        inside_flag_flat_list = []
        approx_flat_list = []
        square_flat_list = []
        for i in range(num_imgs):
            assert len(square_list[i]) == len(inside_flag_list[i])
            inside_flag_flat_list.append(torch.cat(inside_flag_list[i]))
            approx_flat_list.append(torch.cat(approx_list[i]))
            square_flat_list.append(torch.cat(square_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_cls_targets,
         all_bbox_cls_weights, all_bbox_reg_targets, all_bbox_reg_weights,
         pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             approx_flat_list,
             inside_flag_flat_list,
             square_flat_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             sampling=sampling,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_squares)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_squares)
        bbox_cls_targets_list = images_to_levels(all_bbox_cls_targets,
                                                 num_level_squares)
        bbox_cls_weights_list = images_to_levels(all_bbox_cls_weights,
                                                 num_level_squares)
        bbox_reg_targets_list = images_to_levels(all_bbox_reg_targets,
                                                 num_level_squares)
        bbox_reg_weights_list = images_to_levels(all_bbox_reg_weights,
                                                 num_level_squares)
        return (labels_list, label_weights_list, bbox_cls_targets_list,
                bbox_cls_weights_list, bbox_reg_targets_list,
                bbox_reg_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           flat_approxs,
                           inside_flags,
                           flat_squares,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=None,
                           sampling=True,
                           unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_approxs (Tensor): flat approxs of a single image,
                shape (n, 4)
            inside_flags (Tensor): inside flags of a single image,
                shape (n, ).
            flat_squares (Tensor): flat squares of a single image,
                shape (approxs_per_octave * n, 4)
            gt_bboxes (Tensor): Ground truth bboxes of a single image, \
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            sampling (bool): Sample Anchors or not.
            unmap_outputs (bool): unmap outputs or not.

        Returns:
            tuple:

                - labels_list (Tensor): Labels in a single image
                - label_weights (Tensor): Label weights in a single image
                - bbox_cls_targets (Tensor): BBox cls targets in a single image
                - bbox_cls_weights (Tensor): BBox cls weights in a single image
                - bbox_reg_targets (Tensor): BBox reg targets in a single image
                - bbox_reg_weights (Tensor): BBox reg weights in a single image
                - num_total_pos (int): Number of positive samples \
                    in a single image
                - num_total_neg (int): Number of negative samples \
                    in a single image
        """
        if not inside_flags.any():
            return (None, ) * 8
        # assign gt and sample anchors
        expand_inside_flags = inside_flags[:, None].expand(
            -1, self.approxs_per_octave).reshape(-1)
        approxs = flat_approxs[expand_inside_flags, :]
        squares = flat_squares[inside_flags, :]

        assign_result = self.assigner.assign(approxs, squares,
                                             self.approxs_per_octave,
                                             gt_bboxes, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, squares,
                                              gt_bboxes)

        num_valid_squares = squares.shape[0]
        bbox_cls_targets = squares.new_zeros(
            (num_valid_squares, self.side_num * 4))
        bbox_cls_weights = squares.new_zeros(
            (num_valid_squares, self.side_num * 4))
        bbox_reg_targets = squares.new_zeros(
            (num_valid_squares, self.side_num * 4))
        bbox_reg_weights = squares.new_zeros(
            (num_valid_squares, self.side_num * 4))
        labels = squares.new_full((num_valid_squares, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = squares.new_zeros(num_valid_squares, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            (pos_bbox_reg_targets, pos_bbox_reg_weights, pos_bbox_cls_targets,
             pos_bbox_cls_weights) = self.bbox_coder.encode(
                 sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)

            bbox_cls_targets[pos_inds, :] = pos_bbox_cls_targets
            bbox_reg_targets[pos_inds, :] = pos_bbox_reg_targets
            bbox_cls_weights[pos_inds, :] = pos_bbox_cls_weights
            bbox_reg_weights[pos_inds, :] = pos_bbox_reg_weights
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_squares.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_cls_targets = unmap(bbox_cls_targets, num_total_anchors,
                                     inside_flags)
            bbox_cls_weights = unmap(bbox_cls_weights, num_total_anchors,
                                     inside_flags)
            bbox_reg_targets = unmap(bbox_reg_targets, num_total_anchors,
                                     inside_flags)
            bbox_reg_weights = unmap(bbox_reg_weights, num_total_anchors,
                                     inside_flags)
        return (labels, label_weights, bbox_cls_targets, bbox_cls_weights,
                bbox_reg_targets, bbox_reg_weights, pos_inds, neg_inds)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_cls_targets, bbox_cls_weights, bbox_reg_targets,
                    bbox_reg_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_cls_targets = bbox_cls_targets.reshape(-1, self.side_num * 4)
        bbox_cls_weights = bbox_cls_weights.reshape(-1, self.side_num * 4)
        bbox_reg_targets = bbox_reg_targets.reshape(-1, self.side_num * 4)
        bbox_reg_weights = bbox_reg_weights.reshape(-1, self.side_num * 4)
        (bbox_cls_pred, bbox_reg_pred) = bbox_pred
        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(
            -1, self.side_num * 4)
        bbox_reg_pred = bbox_reg_pred.permute(0, 2, 3, 1).reshape(
            -1, self.side_num * 4)
        loss_bbox_cls = self.loss_bbox_cls(
            bbox_cls_pred,
            bbox_cls_targets.long(),
            bbox_cls_weights,
            avg_factor=num_total_samples * 4 * self.side_num)
        loss_bbox_reg = self.loss_bbox_reg(
            bbox_reg_pred,
            bbox_reg_targets,
            bbox_reg_weights,
            avg_factor=num_total_samples * 4 * self.bbox_coder.offset_topk)
        return loss_cls, loss_bbox_cls, loss_bbox_reg

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.approx_anchor_generator.num_levels

        device = cls_scores[0].device

        # get sampled approxes
        approxs_list, inside_flag_list = GuidedAnchorHead.get_sampled_approxs(
            self, featmap_sizes, img_metas, device=device)

        square_list = self.get_anchors(featmap_sizes, img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_target(
            approxs_list,
            inside_flag_list,
            square_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_cls_targets_list,
         bbox_cls_weights_list, bbox_reg_targets_list, bbox_reg_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox_cls, losses_bbox_reg = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_cls_targets_list,
            bbox_cls_weights_list,
            bbox_reg_targets_list,
            bbox_reg_weights_list,
            num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls,
            loss_bbox_cls=losses_bbox_cls,
            loss_bbox_reg=losses_bbox_reg)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        device = cls_scores[0].device
        mlvl_anchors = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_cls_pred_list = [
                bbox_preds[i][0][img_id].detach() for i in range(num_levels)
            ]
            bbox_reg_pred_list = [
                bbox_preds[i][1][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list,
                                               bbox_cls_pred_list,
                                               bbox_reg_pred_list,
                                               mlvl_anchors[img_id], img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_cls_preds,
                          bbox_reg_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_confidences = []
        assert len(cls_scores) == len(bbox_cls_preds) == len(
            bbox_reg_preds) == len(mlvl_anchors)
        for cls_score, bbox_cls_pred, bbox_reg_pred, anchors in zip(
                cls_scores, bbox_cls_preds, bbox_reg_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_cls_pred.size(
            )[-2:] == bbox_reg_pred.size()[-2::]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(
                -1, self.side_num * 4)
            bbox_reg_pred = bbox_reg_pred.permute(1, 2, 0).reshape(
                -1, self.side_num * 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_cls_pred = bbox_cls_pred[topk_inds, :]
                bbox_reg_pred = bbox_reg_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bbox_preds = [
                bbox_cls_pred.contiguous(),
                bbox_reg_pred.contiguous()
            ]
            bboxes, confidences = self.bbox_coder.decode(
                anchors.contiguous(), bbox_preds, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_confidences.append(confidences)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_confidences = torch.cat(mlvl_confidences)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_confidences)
        return det_bboxes, det_labels
