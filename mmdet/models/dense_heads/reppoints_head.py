# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from ..task_modules.prior_generators import MlvlPointGenerator
from ..task_modules.samplers import PseudoSampler
from ..utils import (filter_scores_and_topk, images_to_levels, multi_apply,
                     unmap)
from .anchor_free_head import AnchorFreeHead


@MODELS.register_module()
class RepPointsHead(AnchorFreeHead):
    """RepPoint head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        point_feat_channels (int): Number of channels of points features.
        num_points (int): Number of points.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Sequence[int]): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox_init (:obj:`ConfigDict` or dict): Config of initial points
            loss.
        loss_bbox_refine (:obj:`ConfigDict` or dict): Config of points loss in
            refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 point_feat_channels: int = 256,
                 num_points: int = 9,
                 gradient_mul: float = 0.1,
                 point_strides: Sequence[int] = [8, 16, 32, 64, 128],
                 point_base_scale: int = 4,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 use_grid_points: bool = False,
                 center_init: bool = True,
                 transform_method: str = 'moment',
                 moment_mul: float = 0.01,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='reppoints_cls_out',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.use_grid_points = use_grid_points
        self.center_init = center_init

        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            init_cfg=init_cfg,
            **kwargs)

        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.prior_generator = MlvlPointGenerator(
            self.point_strides, offset=0.)

        if self.train_cfg:
            self.init_assigner = TASK_UTILS.build(
                self.train_cfg['init']['assigner'])
            self.refine_assigner = TASK_UTILS.build(
                self.train_cfg['refine']['assigner'])

            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.loss_bbox_init = MODELS.build(loss_bbox_init)
        self.loss_bbox_refine = MODELS.build(loss_bbox_refine)

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
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        self.reppoints_cls_conv = DeformConv2d(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1,
                                               self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def points2bbox(self, pts: Tensor, y_first: bool = True) -> Tensor:
        """Converting the points set into bounding box.

        Args:
            pts (Tensor): the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first (bool): if y_first=True, the point set is
                represented as [y1, x1, y2, x2 ... yn, xn], otherwise
                the point set is represented as
                [x1, y1, x2, y2 ... xn, yn]. Defaults to True.

        Returns:
            Tensor: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                             dim=1)
        else:
            raise NotImplementedError
        return bbox

    def gen_grid_from_reg(self, reg: Tensor,
                          previous_boxes: Tensor) -> Tuple[Tensor]:
        """Base on the previous bboxes and regression values, we compute the
        regressed bboxes and generate the grids on the bboxes.

        Args:
            reg (Tensor): the regression value to previous bboxes.
            previous_boxes (Tensor): previous bboxes.

        Returns:
            Tuple[Tensor]: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor]:
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x: Tensor) -> Tuple[Tensor]:
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale,
                                      scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        if self.training:
            return cls_out, pts_out_init, pts_out_refine
        else:
            return cls_out, self.points2bbox(pts_out_refine)

    def get_points(self, featmap_sizes: List[Tuple[int]],
                   batch_img_metas: List[dict], device: str) -> tuple:
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(batch_img_metas)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device=device)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def centers_to_bboxes(self, point_list: List[Tensor]) -> List[Tensor]:
        """Get bboxes according to center points.

        Only used in :class:`MaxIoUAssigner`.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list: List[Tensor],
                      pred_list: List[Tensor]) -> List[Tensor]:
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def _get_targets_single(self,
                            flat_proposals: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            gt_instances_ignore: InstanceData,
                            stage: str = 'init',
                            unmap_outputs: bool = True) -> tuple:
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            flat_proposals (Tensor): Multi level points of a image.
            valid_flags (Tensor): Multi level valid flags of a image.
            gt_instances (InstanceData): It usually includes ``bboxes`` and
                ``labels`` attributes.
            gt_instances_ignore (InstanceData): It includes ``bboxes``
                attribute data that is ignored during training and testing.
            stage (str): 'init' or 'refine'. Generate target for
                init stage or refine stage. Defaults to 'init'.
            unmap_outputs (bool): Whether to map outputs back to
                the original set of anchors. Defaults to True.

        Returns:
            tuple:

                - labels (Tensor): Labels of each level.
                - label_weights (Tensor): Label weights of each level.
                - bbox_targets (Tensor): BBox targets of each level.
                - bbox_weights (Tensor): BBox weights of each level.
                - pos_inds (Tensor): positive samples indexes.
                - neg_inds (Tensor): negative samples indexes.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = valid_flags
        if not inside_flags.any():
            raise ValueError(
                'There is no valid proposal inside the image boundary. Please '
                'check the image size.')
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]
        pred_instances = InstanceData(priors=proposals)

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg['init']['pos_weight']
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg['refine']['pos_weight']

        assign_result = assigner.assign(pred_instances, gt_instances,
                                        gt_instances_ignore)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            bbox_gt[pos_inds, :] = sampling_result.pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(
                labels,
                num_total_proposals,
                inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, sampling_result)

    def get_targets(self,
                    proposals_list: List[Tensor],
                    valid_flag_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    stage: str = 'init',
                    unmap_outputs: bool = True,
                    return_sampling_results: bool = False) -> tuple:
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[Tensor]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[Tensor]): Multi level valid flags of each
                image.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            stage (str): 'init' or 'refine'. Generate target for init stage or
                refine stage.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple:

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposals_list (list[Tensor]): Proposals(points/bboxes) of
                  each level.
                - proposal_weights_list (list[Tensor]): Proposal weights of
                  each level.
                - avg_factor (int): Average factor that is used to average
                  the loss. When using sampling method, avg_factor is usually
                  the sum of positive and negative priors. When using
                  `PseudoSampler`, `avg_factor` is usually equal to the number
                  of positive priors.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(batch_img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
             self._get_targets_single,
             proposals_list,
             valid_flag_list,
             batch_gt_instances,
             batch_gt_instances_ignore,
             stage=stage,
             unmap_outputs=unmap_outputs)

        # sampled points of all images
        avg_refactor = sum(
            [results.avg_factor for results in sampling_results_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)
        res = (labels_list, label_weights_list, bbox_gt_list, proposals_list,
               proposal_weights_list, avg_refactor)
        if return_sampling_results:
            res = res + (sampling_results_list, )

        return res

    def loss_by_feat_single(self, cls_score: Tensor, pts_pred_init: Tensor,
                            pts_pred_refine: Tensor, labels: Tensor,
                            label_weights, bbox_gt_init: Tensor,
                            bbox_weights_init: Tensor, bbox_gt_refine: Tensor,
                            bbox_weights_refine: Tensor, stride: int,
                            avg_factor_init: int,
                            avg_factor_refine: int) -> Tuple[Tensor]:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_classes, h_i, w_i).
            pts_pred_init (Tensor): Points of shape
                (batch_size, h_i * w_i, num_points * 2).
            pts_pred_refine (Tensor): Points refined of shape
                (batch_size, h_i * w_i, num_points * 2).
            labels (Tensor): Ground truth class indices with shape
                (batch_size, h_i * w_i).
            label_weights (Tensor): Label weights of shape
                (batch_size, h_i * w_i).
            bbox_gt_init (Tensor): BBox regression targets in the init stage
                of shape (batch_size, h_i * w_i, 4).
            bbox_weights_init (Tensor): BBox regression loss weights in the
                init stage of shape (batch_size, h_i * w_i, 4).
            bbox_gt_refine (Tensor): BBox regression targets in the refine
                stage of shape (batch_size, h_i * w_i, 4).
            bbox_weights_refine (Tensor): BBox regression loss weights in the
                refine stage of shape (batch_size, h_i * w_i, 4).
            stride (int): Point stride.
            avg_factor_init (int): Average factor that is used to average
                the loss in the init stage.
            avg_factor_refine (int): Average factor that is used to average
                the loss in the refine stage.

        Returns:
            Tuple[Tensor]: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score.contiguous()
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor_refine)

        # points loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(
            pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = self.points2bbox(
            pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        normalize_term = self.point_base_scale * stride
        loss_pts_init = self.loss_bbox_init(
            bbox_pred_init / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            avg_factor=avg_factor_init)
        loss_pts_refine = self.loss_bbox_refine(
            bbox_pred_refine / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            avg_factor=avg_factor_refine)
        return loss_cls, loss_pts_init, loss_pts_refine

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        pts_preds_init: List[Tensor],
        pts_preds_refine: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, of shape (batch_size, num_classes, h, w).
            pts_preds_init (list[Tensor]): Points for each scale level, each is
                a 3D-tensor, of shape (batch_size, h_i * w_i, num_points * 2).
            pts_preds_refine (list[Tensor]): Points refined for each scale
                level, each is a 3D-tensor, of shape
                (batch_size, h_i * w_i, num_points * 2).
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
        device = cls_scores[0].device

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       batch_img_metas, device)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)
        if self.train_cfg['init']['assigner']['type'] == 'PointAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            # transform center list to bbox list and
            #   assign target for bbox list
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list
        cls_reg_targets_init = self.get_targets(
            proposals_list=candidate_list,
            valid_flag_list=valid_flag_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            stage='init',
            return_sampling_results=False)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         avg_factor_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       batch_img_metas, device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                bbox_preds_init = self.points2bbox(
                    pts_preds_init[i_lvl].detach())
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat(
                    [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center +
                            bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
            bbox_list.append(bbox)
        cls_reg_targets_refine = self.get_targets(
            proposals_list=bbox_list,
            valid_flag_list=valid_flag_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            stage='refine',
            return_sampling_results=False)
        (labels_list, label_weights_list, bbox_gt_list_refine,
         candidate_list_refine, bbox_weights_list_refine,
         avg_factor_refine) = cls_reg_targets_refine

        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            pts_coordinate_preds_init,
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            self.point_strides,
            avg_factor_init=avg_factor_init,
            avg_factor_refine=avg_factor_refine)
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        return loss_dict_all

    # Same as base_dense_head/_get_bboxes_single except self._bbox_decode
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RepPoints head does not need
                this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (:obj:`ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = self._bbox_decode(priors, bbox_pred,
                                       self.point_strides[level_idx],
                                       img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_decode(self, points: Tensor, bbox_pred: Tensor, stride: int,
                     max_shape: Tuple[int, int]) -> Tensor:
        """Decode the prediction to bounding box.

        Args:
            points (Tensor): shape (h_i * w_i, 2).
            bbox_pred (Tensor): shape (h_i * w_i, 4).
            stride (int): Stride for bbox_pred in different level.
            max_shape (Tuple[int, int]): image shape.

        Returns:
            Tensor: Bounding boxes decoded.
        """
        bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
        bboxes = bbox_pred * stride + bbox_pos_center
        x1 = bboxes[:, 0].clamp(min=0, max=max_shape[1])
        y1 = bboxes[:, 1].clamp(min=0, max=max_shape[0])
        x2 = bboxes[:, 2].clamp(min=0, max=max_shape[1])
        y2 = bboxes[:, 3].clamp(min=0, max=max_shape[0])
        decoded_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return decoded_bboxes
