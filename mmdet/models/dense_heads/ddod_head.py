# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import images_to_levels, multi_apply, unmap
from .anchor_head import AnchorHead

EPS = 1e-12


@MODELS.register_module()
class DDODHead(AnchorHead):
    """Detection Head of `DDOD <https://arxiv.org/abs/2107.02963>`_.

    DDOD head decomposes conjunctions lying in most current one-stage
    detectors via label assignment disentanglement, spatial feature
    disentanglement, and pyramid supervision disentanglement.

    Args:
        num_classes (int): Number of categories excluding the
            background category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): The number of stacked Conv. Defaults to 4.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        use_dcn (bool): Use dcn, Same as ATSS when False. Defaults to True.
        norm_cfg (:obj:`ConfigDict` or dict): Normal config of ddod head.
            Defaults to dict(type='GN', num_groups=32, requires_grad=True).
        loss_iou (:obj:`ConfigDict` or dict): Config of IoU loss. Defaults to
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0).
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 stacked_convs: int = 4,
                 conv_cfg: OptConfigType = None,
                 use_dcn: bool = True,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 loss_iou: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 **kwargs) -> None:
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_dcn = use_dcn
        super().__init__(num_classes, in_channels, **kwargs)

        if self.train_cfg:
            self.cls_assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.reg_assigner = TASK_UTILS.build(
                self.train_cfg['reg_assigner'])
        self.loss_iou = MODELS.build(loss_iou)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
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
                    conv_cfg=dict(type='DCN', deform_groups=1)
                    if i == 0 and self.use_dcn else self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=dict(type='DCN', deform_groups=1)
                    if i == 0 and self.use_dcn else self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.atss_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.atss_iou = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        # we use the global list in loss
        self.cls_num_pos_samples_per_level = [
            0. for _ in range(len(self.prior_generator.strides))
        ]
        self.reg_num_pos_samples_per_level = [
            0. for _ in range(len(self.prior_generator.strides))
        ]

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.atss_reg, std=0.01)
        normal_init(self.atss_iou, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores, bbox predictions,
            and iou predictions.

            - cls_scores (list[Tensor]): Classification scores for all \
            scale levels, each is a 4D-tensor, the channels number is \
            num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all \
            scale levels, each is a 4D-tensor, the channels number is \
            num_base_priors * 4.
            - iou_preds (list[Tensor]): IoU scores for all scale levels, \
            each is a 4D-tensor, the channels number is num_base_priors * 1.
        """
        return multi_apply(self.forward_single, x, self.scales)

    def forward_single(self, x: Tensor, scale: Scale) -> Sequence[Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:

            - cls_score (Tensor): Cls scores for a single scale level \
            the channels number is num_base_priors * num_classes.
            - bbox_pred (Tensor): Box energies / deltas for a single \
            scale level, the channels number is num_base_priors * 4.
            - iou_pred (Tensor): Iou for a single scale level, the \
            channel number is (N, num_base_priors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        iou_pred = self.atss_iou(reg_feat)
        return cls_score, bbox_pred, iou_pred

    def loss_cls_by_feat_single(self, cls_score: Tensor, labels: Tensor,
                                label_weights: Tensor,
                                reweight_factor: List[float],
                                avg_factor: float) -> Tuple[Tensor]:
        """Compute cls loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_base_priors * num_classes, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            reweight_factor (List[float]): Reweight factor for cls and reg
                loss.
            avg_factor (float): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            Tuple[Tensor]: A tuple of loss components.
        """
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        return reweight_factor * loss_cls,

    def loss_reg_by_feat_single(self, anchors: Tensor, bbox_pred: Tensor,
                                iou_pred: Tensor, labels,
                                label_weights: Tensor, bbox_targets: Tensor,
                                bbox_weights: Tensor,
                                reweight_factor: List[float],
                                avg_factor: float) -> Tuple[Tensor, Tensor]:
        """Compute reg loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_base_priors * 4, H, W).
            iou_pred (Tensor): Iou for a single scale level, the
                channel number is (N, num_base_priors * 1, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox weights of all anchors in the
                image with shape (N, 4)
            reweight_factor (List[float]): Reweight factor for cls and reg
                loss.
            avg_factor (float): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.
        Returns:
            Tuple[Tensor, Tensor]: A tuple of loss components.
        """
        anchors = anchors.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1, )
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        iou_targets = label_weights.new_zeros(labels.shape)
        iou_weights = label_weights.new_zeros(labels.shape)
        iou_weights[(bbox_weights.sum(axis=1) > 0).nonzero(
            as_tuple=False)] = 1.

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    &
                    (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]

            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                avg_factor=avg_factor)

            iou_targets[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            loss_iou = self.loss_iou(
                iou_pred, iou_targets, iou_weights, avg_factor=avg_factor)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_iou = iou_pred.sum() * 0

        return reweight_factor * loss_bbox, reweight_factor * loss_iou

    def calc_reweight_factor(self, labels_list: List[Tensor]) -> List[float]:
        """Compute reweight_factor for regression and classification loss."""
        # get pos samples for each level
        bg_class_ind = self.num_classes
        for ii, each_level_label in enumerate(labels_list):
            pos_inds = ((each_level_label >= 0) &
                        (each_level_label < bg_class_ind)).nonzero(
                            as_tuple=False).squeeze(1)
            self.cls_num_pos_samples_per_level[ii] += len(pos_inds)
        # get reweight factor from 1 ~ 2 with bilinear interpolation
        min_pos_samples = min(self.cls_num_pos_samples_per_level)
        max_pos_samples = max(self.cls_num_pos_samples_per_level)
        interval = 1. / (max_pos_samples - min_pos_samples + 1e-10)
        reweight_factor_per_level = []
        for pos_samples in self.cls_num_pos_samples_per_level:
            factor = 2. - (pos_samples - min_pos_samples) * interval
            reweight_factor_per_level.append(factor)
        return reweight_factor_per_level

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            iou_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_base_priors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_base_priors * 4, H, W)
            iou_preds (list[Tensor]): Score factor for all scale level,
                each is a 4D-tensor, has shape (batch_size, 1, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        # calculate common vars for cls and reg assigners at once
        targets_com = self.process_predictions_and_anchors(
            anchor_list, valid_flag_list, cls_scores, bbox_preds,
            batch_img_metas, batch_gt_instances_ignore)
        (anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list,
         bbox_pred_list, batch_gt_instances_ignore) = targets_com

        # classification branch assigner
        cls_targets = self.get_cls_targets(
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            cls_score_list,
            bbox_pred_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (cls_anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()
        avg_factor = max(avg_factor, 1.0)

        reweight_factor_per_level = self.calc_reweight_factor(labels_list)

        cls_losses_cls, = multi_apply(
            self.loss_cls_by_feat_single,
            cls_scores,
            labels_list,
            label_weights_list,
            reweight_factor_per_level,
            avg_factor=avg_factor)

        # regression branch assigner
        reg_targets = self.get_reg_targets(
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            cls_score_list,
            bbox_pred_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        (reg_anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()
        avg_factor = max(avg_factor, 1.0)

        reweight_factor_per_level = self.calc_reweight_factor(labels_list)

        reg_losses_bbox, reg_losses_iou = multi_apply(
            self.loss_reg_by_feat_single,
            reg_anchor_list,
            bbox_preds,
            iou_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            reweight_factor_per_level,
            avg_factor=avg_factor)

        return dict(
            loss_cls=cls_losses_cls,
            loss_bbox=reg_losses_bbox,
            loss_iou=reg_losses_iou)

    def process_predictions_and_anchors(
            self,
            anchor_list: List[List[Tensor]],
            valid_flag_list: List[List[Tensor]],
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> tuple:
        """Compute common vars for regression and classification targets.

        Args:
            anchor_list (List[List[Tensor]]): anchors of each image.
            valid_flag_list (List[List[Tensor]]): Valid flags of each image.
            cls_scores (List[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Return:
            tuple[Tensor]: A tuple of common loss vars.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        anchor_list_ = []
        valid_flag_list_ = []
        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list_.append(torch.cat(anchor_list[i]))
            valid_flag_list_.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None for _ in range(num_imgs)]

        num_levels = len(cls_scores)
        cls_score_list = []
        bbox_pred_list = []

        mlvl_cls_score_list = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_base_priors * self.cls_out_channels)
            for cls_score in cls_scores
        ]
        mlvl_bbox_pred_list = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_base_priors * 4)
            for bbox_pred in bbox_preds
        ]

        for i in range(num_imgs):
            mlvl_cls_tensor_list = [
                mlvl_cls_score_list[j][i] for j in range(num_levels)
            ]
            mlvl_bbox_tensor_list = [
                mlvl_bbox_pred_list[j][i] for j in range(num_levels)
            ]
            cat_mlvl_cls_score = torch.cat(mlvl_cls_tensor_list, dim=0)
            cat_mlvl_bbox_pred = torch.cat(mlvl_bbox_tensor_list, dim=0)
            cls_score_list.append(cat_mlvl_cls_score)
            bbox_pred_list.append(cat_mlvl_bbox_pred)
        return (anchor_list_, valid_flag_list_, num_level_anchors_list,
                cls_score_list, bbox_pred_list, batch_gt_instances_ignore)

    def get_cls_targets(self,
                        anchor_list: List[Tensor],
                        valid_flag_list: List[Tensor],
                        num_level_anchors_list: List[int],
                        cls_score_list: List[Tensor],
                        bbox_pred_list: List[Tensor],
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        batch_gt_instances_ignore: OptInstanceList = None,
                        unmap_outputs: bool = True) -> tuple:
        """Get cls targets for DDOD head.

        This method is almost the same as `AnchorHead.get_targets()`.
        Besides returning the targets as the parent  method does,
        it also returns the anchors as the first element of the
        returned tuple.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            num_level_anchors_list (list[Tensor]): Number of anchors of each
                scale level of all image.
            cls_score_list (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_pred_list (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Return:
            tuple[Tensor]: A tuple of cls targets components.
        """
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
             self._get_targets_single,
             anchor_list,
             valid_flag_list,
             cls_score_list,
             bbox_pred_list,
             num_level_anchors_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs,
             is_cls_assigner=True)
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors_list[0])
        labels_list = images_to_levels(all_labels, num_level_anchors_list[0])
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors_list[0])
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors_list[0])
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors_list[0])
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, avg_factor)

    def get_reg_targets(self,
                        anchor_list: List[Tensor],
                        valid_flag_list: List[Tensor],
                        num_level_anchors_list: List[int],
                        cls_score_list: List[Tensor],
                        bbox_pred_list: List[Tensor],
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        batch_gt_instances_ignore: OptInstanceList = None,
                        unmap_outputs: bool = True) -> tuple:
        """Get reg targets for DDOD head.

        This method is almost the same as `AnchorHead.get_targets()` when
        is_cls_assigner is False. Besides returning the targets as the parent
        method does, it also returns the anchors as the first element of the
        returned tuple.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            num_level_anchors_list (list[Tensor]): Number of anchors of each
                scale level of all image.
            cls_score_list (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_pred_list (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Return:
            tuple[Tensor]: A tuple of reg targets components.
        """
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
             self._get_targets_single,
             anchor_list,
             valid_flag_list,
             cls_score_list,
             bbox_pred_list,
             num_level_anchors_list,
             batch_gt_instances,
             batch_img_metas,
             batch_gt_instances_ignore,
             unmap_outputs=unmap_outputs,
             is_cls_assigner=False)
        # Get `avg_factor` of all images, which calculate in `SamplingResult`.
        # When using sampling method, avg_factor is usually the sum of
        # positive and negative priors. When using `PseudoSampler`,
        # `avg_factor` is usually equal to the number of positive priors.
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors_list[0])
        labels_list = images_to_levels(all_labels, num_level_anchors_list[0])
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors_list[0])
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors_list[0])
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors_list[0])
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, avg_factor)

    def _get_targets_single(self,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            num_level_anchors: List[int],
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True,
                            is_cls_assigner: bool = True) -> tuple:
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_base_priors, 4).
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                shape (num_base_priors,).
            cls_scores (Tensor): Classification scores for all scale
                levels of the image.
            bbox_preds (Tensor): Box energies / deltas for all scale
                levels of the image.
            num_level_anchors (List[int]): Number of anchors of each
                scale level.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.
            is_cls_assigner (bool): Classification or regression.
                Defaults to True.

        Returns:
            tuple: N is the number of total anchors in the image.
            - anchors (Tensor): all anchors in the image with shape (N, 4).
            - labels (Tensor): Labels of all anchors in the image with \
            shape (N, ).
            - label_weights (Tensor): Label weights of all anchor in the \
            image with shape (N, ).
            - bbox_targets (Tensor): BBox targets of all anchors in the \
            image with shape (N, 4).
            - bbox_weights (Tensor): BBox weights of all anchors in the \
            image with shape (N, 4)
            - pos_inds (Tensor): Indices of positive anchor with shape \
            (num_pos, ).
            - neg_inds (Tensor): Indices of negative anchor with shape \
            (num_neg, ).
            - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        bbox_preds_valid = bbox_preds[inside_flags, :]
        cls_scores_valid = cls_scores[inside_flags, :]

        assigner = self.cls_assigner if is_cls_assigner else self.reg_assigner

        # decode prediction out of assigner
        bbox_preds_valid = self.bbox_coder.decode(anchors, bbox_preds_valid)
        pred_instances = InstanceData(
            priors=anchors, bboxes=bbox_preds_valid, scores=cls_scores_valid)

        assign_result = assigner.assign(
            pred_instances=pred_instances,
            num_level_priors=num_level_anchors_inside,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, sampling_result)

    def get_num_level_anchors_inside(self, num_level_anchors: List[int],
                                     inside_flags: Tensor) -> List[int]:
        """Get the anchors of each scale level inside.

        Args:
            num_level_anchors (list[int]): Number of anchors of each
                scale level.
            inside_flags (Tensor): Multi level inside flags of the image,
                which are concatenated into a single tensor of
                shape (num_base_priors,).

        Returns:
            list[int]: Number of anchors of each scale level inside.
        """
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
