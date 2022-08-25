# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, Linear, MaxPool2d
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, OptMultiConfig


@MODELS.register_module()
class MaskIoUHead(BaseModule):
    """Mask IoU Head.

    This head predicts the IoU of predicted masks and corresponding gt masks.

    Args:
        num_convs (int): The number of convolution layers. Defaults to 4.
        num_fcs (int): The number of fully connected layers. Defaults to 2.
        roi_feat_size (int): RoI feature size. Default to 14.
        in_channels (int): The channel number of inputs features.
            Defaults to 256.
        conv_out_channels (int): The feature channels of convolution layers.
            Defaults to 256.
        fc_out_channels (int): The feature channels of fully connected layers.
            Defaults to 1024.
        num_classes (int): Number of categories excluding the background
            category. Defaults to 80.
        loss_iou (:obj:`ConfigDict` or dict): IoU loss.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_convs: int = 4,
        num_fcs: int = 2,
        roi_feat_size: int = 14,
        in_channels: int = 256,
        conv_out_channels: int = 256,
        fc_out_channels: int = 1024,
        num_classes: int = 80,
        loss_iou: ConfigType = dict(type='MSELoss', loss_weight=0.5),
        init_cfg: OptMultiConfig = [
            dict(type='Kaiming', override=dict(name='convs')),
            dict(type='Caffe2Xavier', override=dict(name='fcs')),
            dict(type='Normal', std=0.01, override=dict(name='fc_mask_iou'))
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                # concatenation of mask feature and mask prediction
                in_channels = self.in_channels + 1
            else:
                in_channels = self.conv_out_channels
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(
                Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    stride=stride,
                    padding=1))

        roi_feat_size = _pair(roi_feat_size)
        pooled_area = (roi_feat_size[0] // 2) * (roi_feat_size[1] // 2)
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (
                self.conv_out_channels *
                pooled_area if i == 0 else self.fc_out_channels)
            self.fcs.append(Linear(in_channels, self.fc_out_channels))

        self.fc_mask_iou = Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU()
        self.max_pool = MaxPool2d(2, 2)
        self.loss_iou = MODELS.build(loss_iou)

    def forward(self, mask_feat: Tensor, mask_preds: Tensor) -> Tensor:
        """Forward function.

        Args:
            mask_feat (Tensor): Mask features from upstream models.
            mask_preds (Tensor): Mask predictions from mask head.

        Returns:
            Tensor: Mask IoU predictions.
        """
        mask_preds = mask_preds.sigmoid()
        mask_pred_pooled = self.max_pool(mask_preds.unsqueeze(1))

        x = torch.cat((mask_feat, mask_pred_pooled), 1)

        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        return mask_iou

    def loss_and_target(self, mask_iou_pred: Tensor, mask_preds: Tensor,
                        mask_targets: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        """Calculate the loss and targets of MaskIoUHead.

        Args:
            mask_iou_pred (Tensor): Mask IoU predictions results, has shape
                (num_pos, num_classes)
            mask_preds (Tensor): Mask predictions from mask head, has shape
                (num_pos, mask_size, mask_size).
            mask_targets (Tensor): The ground truth masks assigned with
                predictions, has shape
                (num_pos, mask_size, mask_size).
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It includes ``masks`` inside.
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """
        mask_iou_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            mask_preds=mask_preds,
            mask_targets=mask_targets,
            rcnn_train_cfg=rcnn_train_cfg)

        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss_mask_iou = self.loss_iou(mask_iou_pred[pos_inds],
                                          mask_iou_targets[pos_inds])
        else:
            loss_mask_iou = mask_iou_pred.sum() * 0
        return dict(loss_mask_iou=loss_mask_iou)

    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList, mask_preds: Tensor,
                    mask_targets: Tensor,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        """Compute target of mask IoU.

        Mask IoU target is the IoU of the predicted mask (inside a bbox) and
        the gt mask of corresponding gt mask (the whole instance).
        The intersection area is computed inside the bbox, and the gt mask area
        is computed with two steps, firstly we compute the gt area inside the
        bbox, then divide it by the area ratio of gt area inside the bbox and
        the gt area of the whole instance.

        Args:
            sampling_results (list[:obj:`SamplingResult`]): sampling results.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It includes ``masks`` inside.
            mask_preds (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).
            rcnn_train_cfg (obj:`ConfigDict`): Training config for R-CNN part.

        Returns:
            Tensor: mask iou target (length == num positive).
        """
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.masks for res in batch_gt_instances]

        # compute the area ratio of gt areas inside the proposals and
        # the whole instance
        area_ratios = map(self._get_area_ratio, pos_proposals,
                          pos_assigned_gt_inds, gt_masks)
        area_ratios = torch.cat(list(area_ratios))
        assert mask_targets.size(0) == area_ratios.size(0)

        mask_preds = (mask_preds > rcnn_train_cfg.mask_thr_binary).float()
        mask_pred_areas = mask_preds.sum((-1, -2))

        # mask_preds and mask_targets are binary maps
        overlap_areas = (mask_preds * mask_targets).sum((-1, -2))

        # compute the mask area of the whole instance
        gt_full_areas = mask_targets.sum((-1, -2)) / (area_ratios + 1e-7)

        mask_iou_targets = overlap_areas / (
            mask_pred_areas + gt_full_areas - overlap_areas)
        return mask_iou_targets

    def _get_area_ratio(self, pos_proposals: Tensor,
                        pos_assigned_gt_inds: Tensor,
                        gt_masks: InstanceData) -> Tensor:
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance.

        Args:
            pos_proposals (Tensor): Positive proposals, has shape (num_pos, 4).
            pos_assigned_gt_inds (Tensor): positive proposals assigned ground
                truth index.
            gt_masks (BitmapMask or PolygonMask): Gt masks (the whole instance)
                of each image, with the same shape of the input image.

        Returns:
            Tensor: The area ratio of the gt mask inside the proposal and the
            gt mask of the corresponding instance.
        """
        num_pos = pos_proposals.size(0)
        if num_pos > 0:
            area_ratios = []
            proposals_np = pos_proposals.cpu().numpy()
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            # compute mask areas of gt instances (batch processing for speedup)
            gt_instance_mask_area = gt_masks.areas
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]

                # crop the gt mask inside the proposal
                bbox = proposals_np[i, :].astype(np.int32)
                gt_mask_in_proposal = gt_mask.crop(bbox)

                ratio = gt_mask_in_proposal.areas[0] / (
                    gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-7)
                area_ratios.append(ratio)
            area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(
                pos_proposals.device)
        else:
            area_ratios = pos_proposals.new_zeros((0, ))
        return area_ratios

    def predict_by_feat(self, mask_iou_preds: Tuple[Tensor],
                        results_list: InstanceList) -> InstanceList:
        """Predict the mask iou and calculate it into ``results.scores``.

        Args:
            mask_iou_preds (Tensor): Mask IoU predictions results, has shape
                (num_proposals, num_classes)
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert len(mask_iou_preds) == len(results_list)
        for results, mask_iou_pred in zip(results_list, mask_iou_preds):
            labels = results.labels
            scores = results.scores
            results.scores = scores * mask_iou_pred[range(labels.size(0)),
                                                    labels]
        return results_list
