# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptConfigType


@MODELS.register_module()
class GridHead(BaseModule):
    """Implementation of `Grid Head <https://arxiv.org/abs/1811.12030>`_

    Args:
        grid_points (int): The number of grid points. Defaults to 9.
        num_convs (int): The number of convolution layers. Defaults to 8.
        roi_feat_size (int): RoI feature size. Default to 14.
        in_channels (int): The channel number of inputs features.
            Defaults to 256.
        conv_kernel_size (int): The kernel size of convolution layers.
            Defaults to 3.
        point_feat_channels (int): The number of channels of each point
            features. Defaults to 64.
        class_agnostic (bool): Whether use class agnostic classification.
            If so, the output channels of logits will be 1. Defaults to False.
        loss_grid (:obj:`ConfigDict` or dict): Config of grid loss.
        conv_cfg (:obj:`ConfigDict` or dict, optional) dictionary to
            construct and config conv layer.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """

    def __init__(
        self,
        grid_points: int = 9,
        num_convs: int = 8,
        roi_feat_size: int = 14,
        in_channels: int = 256,
        conv_kernel_size: int = 3,
        point_feat_channels: int = 64,
        deconv_kernel_size: int = 4,
        class_agnostic: bool = False,
        loss_grid: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=15),
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='GN', num_groups=36),
        init_cfg: MultiConfig = [
            dict(type='Kaiming', layer=['Conv2d', 'Linear']),
            dict(
                type='Normal',
                layer='ConvTranspose2d',
                std=0.001,
                override=dict(
                    type='Normal',
                    name='deconv2',
                    std=0.001,
                    bias=-np.log(0.99 / 0.01)))
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.grid_points = grid_points
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.point_feat_channels = point_feat_channels
        self.conv_out_channels = self.point_feat_channels * self.grid_points
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if isinstance(norm_cfg, dict) and norm_cfg['type'] == 'GN':
            assert self.conv_out_channels % norm_cfg['num_groups'] == 0

        assert self.grid_points >= 4
        self.grid_size = int(np.sqrt(self.grid_points))
        if self.grid_size * self.grid_size != self.grid_points:
            raise ValueError('grid_points must be a square number')

        # the predicted heatmap is half of whole_map_size
        if not isinstance(self.roi_feat_size, int):
            raise ValueError('Only square RoIs are supporeted in Grid R-CNN')
        self.whole_map_size = self.roi_feat_size * 4

        # compute point-wise sub-regions
        self.sub_regions = self.calc_sub_regions()

        self.convs = []
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            stride = 2 if i == 0 else 1
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=True))
        self.convs = nn.Sequential(*self.convs)

        self.deconv1 = nn.ConvTranspose2d(
            self.conv_out_channels,
            self.conv_out_channels,
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=grid_points)
        self.norm1 = nn.GroupNorm(grid_points, self.conv_out_channels)
        self.deconv2 = nn.ConvTranspose2d(
            self.conv_out_channels,
            grid_points,
            kernel_size=deconv_kernel_size,
            stride=2,
            padding=(deconv_kernel_size - 2) // 2,
            groups=grid_points)

        # find the 4-neighbor of each grid point
        self.neighbor_points = []
        grid_size = self.grid_size
        for i in range(grid_size):  # i-th column
            for j in range(grid_size):  # j-th row
                neighbors = []
                if i > 0:  # left: (i - 1, j)
                    neighbors.append((i - 1) * grid_size + j)
                if j > 0:  # up: (i, j - 1)
                    neighbors.append(i * grid_size + j - 1)
                if j < grid_size - 1:  # down: (i, j + 1)
                    neighbors.append(i * grid_size + j + 1)
                if i < grid_size - 1:  # right: (i + 1, j)
                    neighbors.append((i + 1) * grid_size + j)
                self.neighbor_points.append(tuple(neighbors))
        # total edges in the grid
        self.num_edges = sum([len(p) for p in self.neighbor_points])

        self.forder_trans = nn.ModuleList()  # first-order feature transition
        self.sorder_trans = nn.ModuleList()  # second-order feature transition
        for neighbors in self.neighbor_points:
            fo_trans = nn.ModuleList()
            so_trans = nn.ModuleList()
            for _ in range(len(neighbors)):
                # each transition module consists of a 5x5 depth-wise conv and
                # 1x1 conv.
                fo_trans.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.point_feat_channels,
                            self.point_feat_channels,
                            5,
                            stride=1,
                            padding=2,
                            groups=self.point_feat_channels),
                        nn.Conv2d(self.point_feat_channels,
                                  self.point_feat_channels, 1)))
                so_trans.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.point_feat_channels,
                            self.point_feat_channels,
                            5,
                            1,
                            2,
                            groups=self.point_feat_channels),
                        nn.Conv2d(self.point_feat_channels,
                                  self.point_feat_channels, 1)))
            self.forder_trans.append(fo_trans)
            self.sorder_trans.append(so_trans)

        self.loss_grid = MODELS.build(loss_grid)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """forward function of ``GridHead``.

        Args:
            x (Tensor): RoI features, has shape
                (num_rois, num_channels, roi_feat_size, roi_feat_size).

        Returns:
            Dict[str, Tensor]: Return a dict including fused and unfused
            heatmap.
        """
        assert x.shape[-1] == x.shape[-2] == self.roi_feat_size
        # RoI feature transformation, downsample 2x
        x = self.convs(x)

        c = self.point_feat_channels
        # first-order fusion
        x_fo = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_fo[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_fo[i] = x_fo[i] + self.forder_trans[i][j](
                    x[:, point_idx * c:(point_idx + 1) * c])

        # second-order fusion
        x_so = [None for _ in range(self.grid_points)]
        for i, points in enumerate(self.neighbor_points):
            x_so[i] = x[:, i * c:(i + 1) * c]
            for j, point_idx in enumerate(points):
                x_so[i] = x_so[i] + self.sorder_trans[i][j](x_fo[point_idx])

        # predicted heatmap with fused features
        x2 = torch.cat(x_so, dim=1)
        x2 = self.deconv1(x2)
        x2 = F.relu(self.norm1(x2), inplace=True)
        heatmap = self.deconv2(x2)

        # predicted heatmap with original features (applicable during training)
        if self.training:
            x1 = x
            x1 = self.deconv1(x1)
            x1 = F.relu(self.norm1(x1), inplace=True)
            heatmap_unfused = self.deconv2(x1)
        else:
            heatmap_unfused = heatmap

        return dict(fused=heatmap, unfused=heatmap_unfused)

    def calc_sub_regions(self) -> List[Tuple[float]]:
        """Compute point specific representation regions.

        See `Grid R-CNN Plus <https://arxiv.org/abs/1906.05688>`_ for details.
        """
        # to make it consistent with the original implementation, half_size
        # is computed as 2 * quarter_size, which is smaller
        half_size = self.whole_map_size // 4 * 2
        sub_regions = []
        for i in range(self.grid_points):
            x_idx = i // self.grid_size
            y_idx = i % self.grid_size
            if x_idx == 0:
                sub_x1 = 0
            elif x_idx == self.grid_size - 1:
                sub_x1 = half_size
            else:
                ratio = x_idx / (self.grid_size - 1) - 0.25
                sub_x1 = max(int(ratio * self.whole_map_size), 0)

            if y_idx == 0:
                sub_y1 = 0
            elif y_idx == self.grid_size - 1:
                sub_y1 = half_size
            else:
                ratio = y_idx / (self.grid_size - 1) - 0.25
                sub_y1 = max(int(ratio * self.whole_map_size), 0)
            sub_regions.append(
                (sub_x1, sub_y1, sub_x1 + half_size, sub_y1 + half_size))
        return sub_regions

    def get_targets(self, sampling_results: List[SamplingResult],
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.".

        Args:
            sampling_results (List[:obj:`SamplingResult`]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (:obj:`ConfigDict`): `train_cfg` of RCNN.

        Returns:
            Tensor: Grid heatmap targets.
        """
        # mix all samples (across images) together.
        pos_bboxes = torch.cat([res.pos_bboxes for res in sampling_results],
                               dim=0).cpu()
        pos_gt_bboxes = torch.cat(
            [res.pos_gt_bboxes for res in sampling_results], dim=0).cpu()
        assert pos_bboxes.shape == pos_gt_bboxes.shape

        # expand pos_bboxes to 2x of original size
        x1 = pos_bboxes[:, 0] - (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
        y1 = pos_bboxes[:, 1] - (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
        x2 = pos_bboxes[:, 2] + (pos_bboxes[:, 2] - pos_bboxes[:, 0]) / 2
        y2 = pos_bboxes[:, 3] + (pos_bboxes[:, 3] - pos_bboxes[:, 1]) / 2
        pos_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        pos_bbox_ws = (pos_bboxes[:, 2] - pos_bboxes[:, 0]).unsqueeze(-1)
        pos_bbox_hs = (pos_bboxes[:, 3] - pos_bboxes[:, 1]).unsqueeze(-1)

        num_rois = pos_bboxes.shape[0]
        map_size = self.whole_map_size
        # this is not the final target shape
        targets = torch.zeros((num_rois, self.grid_points, map_size, map_size),
                              dtype=torch.float)

        # pre-compute interpolation factors for all grid points.
        # the first item is the factor of x-dim, and the second is y-dim.
        # for a 9-point grid, factors are like (1, 0), (0.5, 0.5), (0, 1)
        factors = []
        for j in range(self.grid_points):
            x_idx = j // self.grid_size
            y_idx = j % self.grid_size
            factors.append((1 - x_idx / (self.grid_size - 1),
                            1 - y_idx / (self.grid_size - 1)))

        radius = rcnn_train_cfg.pos_radius
        radius2 = radius**2
        for i in range(num_rois):
            # ignore small bboxes
            if (pos_bbox_ws[i] <= self.grid_size
                    or pos_bbox_hs[i] <= self.grid_size):
                continue
            # for each grid point, mark a small circle as positive
            for j in range(self.grid_points):
                factor_x, factor_y = factors[j]
                gridpoint_x = factor_x * pos_gt_bboxes[i, 0] + (
                    1 - factor_x) * pos_gt_bboxes[i, 2]
                gridpoint_y = factor_y * pos_gt_bboxes[i, 1] + (
                    1 - factor_y) * pos_gt_bboxes[i, 3]

                cx = int((gridpoint_x - pos_bboxes[i, 0]) / pos_bbox_ws[i] *
                         map_size)
                cy = int((gridpoint_y - pos_bboxes[i, 1]) / pos_bbox_hs[i] *
                         map_size)

                for x in range(cx - radius, cx + radius + 1):
                    for y in range(cy - radius, cy + radius + 1):
                        if x >= 0 and x < map_size and y >= 0 and y < map_size:
                            if (x - cx)**2 + (y - cy)**2 <= radius2:
                                targets[i, j, y, x] = 1
        # reduce the target heatmap size by a half
        # proposed in Grid R-CNN Plus (https://arxiv.org/abs/1906.05688).
        sub_targets = []
        for i in range(self.grid_points):
            sub_x1, sub_y1, sub_x2, sub_y2 = self.sub_regions[i]
            sub_targets.append(targets[:, [i], sub_y1:sub_y2, sub_x1:sub_x2])
        sub_targets = torch.cat(sub_targets, dim=1)
        sub_targets = sub_targets.to(sampling_results[0].pos_bboxes.device)
        return sub_targets

    def loss(self, grid_pred: Tensor, sample_idx: Tensor,
             sampling_results: List[SamplingResult],
             rcnn_train_cfg: ConfigDict) -> dict:
        """Calculate the loss based on the features extracted by the grid head.

        Args:
            grid_pred (dict[str, Tensor]): Outputs of grid_head forward.
            sample_idx (Tensor): The sampling index of ``grid_pred``.
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss and targets components.
        """
        grid_targets = self.get_targets(sampling_results, rcnn_train_cfg)
        grid_targets = grid_targets[sample_idx]

        loss_fused = self.loss_grid(grid_pred['fused'], grid_targets)
        loss_unfused = self.loss_grid(grid_pred['unfused'], grid_targets)
        loss_grid = loss_fused + loss_unfused
        return dict(loss_grid=loss_grid)

    def predict_by_feat(self,
                        grid_preds: Dict[str, Tensor],
                        results_list: List[InstanceData],
                        batch_img_metas: List[dict],
                        rescale: bool = False) -> InstanceList:
        """Adjust the predicted bboxes from bbox head.

        Args:
            grid_preds (dict[str, Tensor]): dictionary outputted by forward
                function.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            batch_img_metas (list[dict]): List of image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape \
            (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4), the last \
            dimension 4 arrange as (x1, y1, x2, y2).
        """
        num_roi_per_img = tuple(res.bboxes.size(0) for res in results_list)
        grid_preds = {
            k: v.split(num_roi_per_img, 0)
            for k, v in grid_preds.items()
        }

        for i, results in enumerate(results_list):
            if len(results) != 0:
                bboxes = self._predict_by_feat_single(
                    grid_pred=grid_preds['fused'][i],
                    bboxes=results.bboxes,
                    img_meta=batch_img_metas[i],
                    rescale=rescale)
                results.bboxes = bboxes
        return results_list

    def _predict_by_feat_single(self,
                                grid_pred: Tensor,
                                bboxes: Tensor,
                                img_meta: dict,
                                rescale: bool = False) -> Tensor:
        """Adjust ``bboxes`` according to ``grid_pred``.

        Args:
            grid_pred (Tensor): Grid fused heatmap.
            bboxes (Tensor): Predicted bboxes, has shape (n, 4)
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            Tensor: adjusted bboxes.
        """
        assert bboxes.size(0) == grid_pred.size(0)
        grid_pred = grid_pred.sigmoid()

        R, c, h, w = grid_pred.shape
        half_size = self.whole_map_size // 4 * 2
        assert h == w == half_size
        assert c == self.grid_points

        # find the point with max scores in the half-sized heatmap
        grid_pred = grid_pred.view(R * c, h * w)
        pred_scores, pred_position = grid_pred.max(dim=1)
        xs = pred_position % w
        ys = pred_position // w

        # get the position in the whole heatmap instead of half-sized heatmap
        for i in range(self.grid_points):
            xs[i::self.grid_points] += self.sub_regions[i][0]
            ys[i::self.grid_points] += self.sub_regions[i][1]

        # reshape to (num_rois, grid_points)
        pred_scores, xs, ys = tuple(
            map(lambda x: x.view(R, c), [pred_scores, xs, ys]))

        # get expanded pos_bboxes
        widths = (bboxes[:, 2] - bboxes[:, 0]).unsqueeze(-1)
        heights = (bboxes[:, 3] - bboxes[:, 1]).unsqueeze(-1)
        x1 = (bboxes[:, 0, None] - widths / 2)
        y1 = (bboxes[:, 1, None] - heights / 2)
        # map the grid point to the absolute coordinates
        abs_xs = (xs.float() + 0.5) / w * widths + x1
        abs_ys = (ys.float() + 0.5) / h * heights + y1

        # get the grid points indices that fall on the bbox boundaries
        x1_inds = [i for i in range(self.grid_size)]
        y1_inds = [i * self.grid_size for i in range(self.grid_size)]
        x2_inds = [
            self.grid_points - self.grid_size + i
            for i in range(self.grid_size)
        ]
        y2_inds = [(i + 1) * self.grid_size - 1 for i in range(self.grid_size)]

        # voting of all grid points on some boundary
        bboxes_x1 = (abs_xs[:, x1_inds] * pred_scores[:, x1_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, x1_inds].sum(dim=1, keepdim=True))
        bboxes_y1 = (abs_ys[:, y1_inds] * pred_scores[:, y1_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, y1_inds].sum(dim=1, keepdim=True))
        bboxes_x2 = (abs_xs[:, x2_inds] * pred_scores[:, x2_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, x2_inds].sum(dim=1, keepdim=True))
        bboxes_y2 = (abs_ys[:, y2_inds] * pred_scores[:, y2_inds]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, y2_inds].sum(dim=1, keepdim=True))

        bboxes = torch.cat([bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2], dim=1)
        bboxes[:, [0, 2]].clamp_(min=0, max=img_meta['img_shape'][1])
        bboxes[:, [1, 3]].clamp_(min=0, max=img_meta['img_shape'][0])

        if rescale:
            assert img_meta.get('scale_factor') is not None
            bboxes /= bboxes.new_tensor(img_meta['scale_factor']).repeat(
                (1, 2))

        return bboxes
