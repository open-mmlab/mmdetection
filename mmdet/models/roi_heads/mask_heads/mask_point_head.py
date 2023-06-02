# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend/point_head/point_head.py  # noqa

from typing import List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import point_sample, rel_roi_point_to_rel_img_point
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import (get_uncertain_point_coords_with_randomness,
                                get_uncertainty)
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptConfigType


@MODELS.register_module()
class MaskPointHead(BaseModule):
    """A mask point head use in PointRend.

    ``MaskPointHead`` use shared multi-layer perceptron (equivalent to
    nn.Conv1d) to predict the logit of input points. The fine-grained feature
    and coarse feature will be concatenate together for predication.

    Args:
        num_fcs (int): Number of fc layers in the head. Defaults to 3.
        in_channels (int): Number of input channels. Defaults to 256.
        fc_channels (int): Number of fc channels. Defaults to 256.
        num_classes (int): Number of classes for logits. Defaults to 80.
        class_agnostic (bool): Whether use class agnostic classification.
            If so, the output channels of logits will be 1. Defaults to False.
        coarse_pred_each_layer (bool): Whether concatenate coarse feature with
            the output of each fc layer. Defaults to True.
        conv_cfg (:obj:`ConfigDict` or dict): Dictionary to construct
            and config conv layer. Defaults to dict(type='Conv1d')).
        norm_cfg (:obj:`ConfigDict` or dict, optional): Dictionary to construct
            and config norm layer. Defaults to None.
        loss_point (:obj:`ConfigDict` or dict): Dictionary to construct and
            config loss layer of point head. Defaults to
            dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes: int,
        num_fcs: int = 3,
        in_channels: int = 256,
        fc_channels: int = 256,
        class_agnostic: bool = False,
        coarse_pred_each_layer: bool = True,
        conv_cfg: ConfigType = dict(type='Conv1d'),
        norm_cfg: OptConfigType = None,
        act_cfg: ConfigType = dict(type='ReLU'),
        loss_point: ConfigType = dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        init_cfg: MultiConfig = dict(
            type='Normal', std=0.001, override=dict(name='fc_logits'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.coarse_pred_each_layer = coarse_pred_each_layer
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_point = MODELS.build(loss_point)

        fc_in_channels = in_channels + num_classes
        self.fcs = nn.ModuleList()
        for _ in range(num_fcs):
            fc = ConvModule(
                fc_in_channels,
                fc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += num_classes if self.coarse_pred_each_layer else 0

        out_channels = 1 if self.class_agnostic else self.num_classes
        self.fc_logits = nn.Conv1d(
            fc_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, fine_grained_feats: Tensor,
                coarse_feats: Tensor) -> Tensor:
        """Classify each point base on fine grained and coarse feats.

        Args:
            fine_grained_feats (Tensor): Fine grained feature sampled from FPN,
                shape (num_rois, in_channels, num_points).
            coarse_feats (Tensor): Coarse feature sampled from CoarseMaskHead,
                shape (num_rois, num_classes, num_points).

        Returns:
            Tensor: Point classification results,
            shape (num_rois, num_class, num_points).
        """

        x = torch.cat([fine_grained_feats, coarse_feats], dim=1)
        for fc in self.fcs:
            x = fc(x)
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_feats), dim=1)
        return self.fc_logits(x)

    def get_targets(self, rois: Tensor, rel_roi_points: Tensor,
                    sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    cfg: ConfigType) -> Tensor:
        """Get training targets of MaskPointHead for all images.

        Args:
            rois (Tensor): Region of Interest, shape (num_rois, 5).
            rel_roi_points (Tensor): Points coordinates relative to RoI, shape
                (num_rois, num_points, 2).
            sampling_results (:obj:`SamplingResult`): Sampling result after
                sampling and assignment.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            cfg (obj:`ConfigDict` or dict): Training cfg.

        Returns:
            Tensor: Point target, shape (num_rois, num_points).
        """

        num_imgs = len(sampling_results)
        rois_list = []
        rel_roi_points_list = []
        for batch_ind in range(num_imgs):
            inds = (rois[:, 0] == batch_ind)
            rois_list.append(rois[inds])
            rel_roi_points_list.append(rel_roi_points[inds])
        pos_assigned_gt_inds_list = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        cfg_list = [cfg for _ in range(num_imgs)]

        point_targets = map(self._get_targets_single, rois_list,
                            rel_roi_points_list, pos_assigned_gt_inds_list,
                            batch_gt_instances, cfg_list)
        point_targets = list(point_targets)

        if len(point_targets) > 0:
            point_targets = torch.cat(point_targets)

        return point_targets

    def _get_targets_single(self, rois: Tensor, rel_roi_points: Tensor,
                            pos_assigned_gt_inds: Tensor,
                            gt_instances: InstanceData,
                            cfg: ConfigType) -> Tensor:
        """Get training target of MaskPointHead for each image."""
        num_pos = rois.size(0)
        num_points = cfg.num_points
        if num_pos > 0:
            gt_masks_th = (
                gt_instances.masks.to_tensor(rois.dtype,
                                             rois.device).index_select(
                                                 0, pos_assigned_gt_inds))
            gt_masks_th = gt_masks_th.unsqueeze(1)
            rel_img_points = rel_roi_point_to_rel_img_point(
                rois, rel_roi_points, gt_masks_th)
            point_targets = point_sample(gt_masks_th,
                                         rel_img_points).squeeze(1)
        else:
            point_targets = rois.new_zeros((0, num_points))
        return point_targets

    def loss_and_target(self, point_pred: Tensor, rel_roi_points: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        cfg: ConfigType) -> dict:
        """Calculate loss for MaskPointHead.

        Args:
            point_pred (Tensor): Point predication result, shape
                (num_rois, num_classes, num_points).
            rel_roi_points (Tensor): Points coordinates relative to RoI, shape
                (num_rois, num_points, 2).
             sampling_results (:obj:`SamplingResult`): Sampling result after
                sampling and assignment.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            cfg (obj:`ConfigDict` or dict): Training cfg.

        Returns:
            dict: a dictionary of point loss and point target.
        """
        rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        point_target = self.get_targets(rois, rel_roi_points, sampling_results,
                                        batch_gt_instances, cfg)
        if self.class_agnostic:
            loss_point = self.loss_point(point_pred, point_target,
                                         torch.zeros_like(pos_labels))
        else:
            loss_point = self.loss_point(point_pred, point_target, pos_labels)

        return dict(loss_point=loss_point, point_target=point_target)

    def get_roi_rel_points_train(self, mask_preds: Tensor, labels: Tensor,
                                 cfg: ConfigType) -> Tensor:
        """Get ``num_points`` most uncertain points with random points during
        train.

        Sample points in [0, 1] x [0, 1] coordinate space based on their
        uncertainty. The uncertainties are calculated for each point using
        '_get_uncertainty()' function that takes point's logit prediction as
        input.

        Args:
            mask_preds (Tensor): A tensor of shape (num_rois, num_classes,
                mask_height, mask_width) for class-specific or class-agnostic
                prediction.
            labels (Tensor): The ground truth class for each instance.
            cfg (:obj:`ConfigDict` or dict): Training config of point head.

        Returns:
            point_coords (Tensor): A tensor of shape (num_rois, num_points, 2)
            that contains the coordinates sampled points.
        """
        point_coords = get_uncertain_point_coords_with_randomness(
            mask_preds, labels, cfg.num_points, cfg.oversample_ratio,
            cfg.importance_sample_ratio)
        return point_coords

    def get_roi_rel_points_test(self, mask_preds: Tensor, label_preds: Tensor,
                                cfg: ConfigType) -> Tuple[Tensor, Tensor]:
        """Get ``num_points`` most uncertain points during test.

        Args:
            mask_preds (Tensor): A tensor of shape (num_rois, num_classes,
                mask_height, mask_width) for class-specific or class-agnostic
                prediction.
            label_preds (Tensor): The predication class for each instance.
            cfg (:obj:`ConfigDict` or dict): Testing config of point head.

        Returns:
            tuple:

            - point_indices (Tensor): A tensor of shape (num_rois, num_points)
              that contains indices from [0, mask_height x mask_width) of the
              most uncertain points.
            - point_coords (Tensor): A tensor of shape (num_rois, num_points,
              2) that contains [0, 1] x [0, 1] normalized coordinates of the
              most uncertain points from the [mask_height, mask_width] grid.
        """
        num_points = cfg.subdivision_num_points
        uncertainty_map = get_uncertainty(mask_preds, label_preds)
        num_rois, _, mask_height, mask_width = uncertainty_map.shape

        # During ONNX exporting, the type of each elements of 'shape' is
        # `Tensor(float)`, while it is `float` during PyTorch inference.
        if isinstance(mask_height, torch.Tensor):
            h_step = 1.0 / mask_height.float()
            w_step = 1.0 / mask_width.float()
        else:
            h_step = 1.0 / mask_height
            w_step = 1.0 / mask_width
        # cast to int to avoid dynamic K for TopK op in ONNX
        mask_size = int(mask_height * mask_width)
        uncertainty_map = uncertainty_map.view(num_rois, mask_size)
        num_points = min(mask_size, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]
        xs = w_step / 2.0 + (point_indices % mask_width).float() * w_step
        ys = h_step / 2.0 + (point_indices // mask_width).float() * h_step
        point_coords = torch.stack([xs, ys], dim=2)
        return point_indices, point_coords
